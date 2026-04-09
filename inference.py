"""
CreditLens — Inference Script (contest-compliant v4)

Emits EXACTLY the stdout format required by the validator:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Environment variables (set in HuggingFace Space secrets):
  API_BASE_URL  CreditLens server URL   (default: http://localhost:7860)
  MODEL_NAME    LLM model identifier    (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      HuggingFace API key     (required for LLM calls)
  LLM_BASE_URL  LLM router base URL     (default: https://router.huggingface.co/v1)
"""
from __future__ import annotations
import json, os, re, sys, time
from typing import List, Optional
import httpx
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
BENCHMARK    = "creditlens"
MAX_STEPS    = 25
SUCCESS_THRESHOLD = 0.5

# ── Exact stdout format ────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    a = str(action).replace(" ","_").replace("\n","").replace("\r","")[:80]
    e = error if error else "null"
    print(f"[STEP] step={step} action={a} reward={reward:.2f} done={str(done).lower()} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)

# ── REST client ────────────────────────────────────────────────────────────

class CreditLensClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=30.0)
    def health(self):
        try:
            return self._http.get(f"{self.base_url}/health", timeout=5.0).status_code == 200
        except Exception:
            return False
    def reset(self, task_id="easy", seed=42):
        r = self._http.post(f"{self.base_url}/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()
    def step(self, task_id, action):
        r = self._http.post(f"{self.base_url}/step", json={"task_id": task_id, "action": action})
        try:
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"reward":0.0,"done":True,"error":str(e),"observation":None}
    def grade(self, task_id):
        r = self._http.get(f"{self.base_url}/grade", params={"task_id": task_id})
        r.raise_for_status()
        return r.json()

# ── Rule-based fallback ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior AI loan underwriter. Decide on each applicant.
Respond ONLY with JSON — no other text:
{"action_type":"APPROVE"|"REJECT"|"COUNTER"|"REQUEST_INFO","params":{...},"reasoning":"one sentence"}
Rules (stop at first match):
1. fraud_ring_score>0.70 OR (shared_phone=true AND shared_employer=true) → REJECT FRAUD_SUSPECTED
2. xgb_default_prob>0.62 → REJECT HIGH_DTI
3. ECL_usage>80% AND xgb>0.15 → REJECT HIGH_DTI
4. ECL_usage>60% AND xgb>0.35 → REJECT HIGH_DTI
5. macro_shock=true AND xgb>0.47 → REJECT HIGH_DTI
6. macro_shock=true AND xgb>0.30 → COUNTER revised_amount_fraction=0.75
7. fico<580 → REJECT LOW_CREDIT_SCORE
8. dti>0.50 → REJECT HIGH_DTI
9. xgb>0.40 OR fico<620 → COUNTER revised_amount_fraction=0.75
10. Otherwise → APPROVE amount_fraction=1.0"""

def _rule_based(obs):
    fico=obs.get("fico_score",650); xgb=obs.get("xgb_default_prob",0.3)
    dti=obs.get("dti_ratio",0.35); fraud=obs.get("fraud_ring_score",0.0)
    shock=obs.get("macro_shock_active",False); sp=obs.get("shared_phone",False)
    se=obs.get("shared_employer_id",False); ecl=obs.get("portfolio_ecl",0.0)
    budget=obs.get("portfolio_ecl_budget",0.05); app_id=obs.get("applicant_id","unknown")
    ref_r=obs.get("approval_rate_reference",0.5); prot_r=obs.get("approval_rate_protected",0.5)
    group=obs.get("demographic_group","group_a")
    ecl_use=ecl/max(budget,1e-8); shock_adj=0.15 if shock else 0.0
    gap=ref_r-prot_r; is_prot=group in ("group_b","group_c")
    def rej(r): return {"action_type":"REJECT","applicant_id":app_id,"params":{"reason_code":r},"reasoning":r}
    if fraud>0.70 or (sp and se): return rej("FRAUD_SUSPECTED")
    xgb_cut=0.62-shock_adj
    if xgb>xgb_cut: return rej("HIGH_DTI")
    et=1.1
    if ecl_use>0.95: et=0.10
    elif ecl_use>0.80: et=0.15
    elif ecl_use>0.60: et=0.35
    et-=shock_adj
    if xgb>et: return rej("HIGH_DTI")
    if gap>0.12 and is_prot and xgb<0.60 and fico>=600 and dti<0.50 and fraud<0.50:
        return {"action_type":"APPROVE","applicant_id":app_id,"params":{"amount_fraction":0.90},"reasoning":"fairness rescue"}
    if gap<-0.12 and is_prot and xgb>0.35:
        return {"action_type":"COUNTER","applicant_id":app_id,"params":{"revised_amount_fraction":0.75,"revised_rate_delta":1.5},"reasoning":"fairness rebalance"}
    if shock and xgb>0.30:
        return {"action_type":"COUNTER","applicant_id":app_id,"params":{"revised_amount_fraction":0.75,"revised_rate_delta":2.0},"reasoning":"post-shock counter"}
    if fico<580: return rej("LOW_CREDIT_SCORE")
    if dti>0.50: return rej("HIGH_DTI")
    if xgb>0.40 or fico<620 or dti>0.40:
        frac=0.70 if xgb>0.50 else 0.82
        return {"action_type":"COUNTER","applicant_id":app_id,"params":{"revised_amount_fraction":frac,"revised_rate_delta":1.5},"reasoning":"moderate risk counter"}
    return {"action_type":"APPROVE","applicant_id":app_id,"params":{"amount_fraction":1.0},"reasoning":"creditworthy"}

def _llm_action(llm, obs):
    try:
        ecl_use=obs.get("portfolio_ecl",0)/max(obs.get("portfolio_ecl_budget",0.05),1e-8)
        summary=(f"id={obs.get('applicant_id')} fico={obs.get('fico_score')} "
                 f"xgb={obs.get('xgb_default_prob',0):.2f} dti={obs.get('dti_ratio',0):.2f} "
                 f"fraud={obs.get('fraud_ring_score',0):.2f} shock={obs.get('macro_shock_active',False)} "
                 f"ecl_pct={ecl_use:.0%} group={obs.get('demographic_group','group_a')} "
                 f"ref={obs.get('approval_rate_reference',0.5):.0%} prot={obs.get('approval_rate_protected',0.5):.0%} "
                 f"sp={obs.get('shared_phone',False)} se={obs.get('shared_employer_id',False)}")
        comp=llm.chat.completions.create(model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":summary}],
            temperature=0.1, max_tokens=200)
        text=(comp.choices[0].message.content or "").strip()
        text=re.sub(r"```(?:json)?|```","",text).strip().replace("\n"," ").replace("\r"," ")
        m=re.search(r"\{.*\}",text,re.DOTALL)
        if m:
            d=json.loads(m.group())
            d.setdefault("applicant_id",obs.get("applicant_id","unknown"))
            if d.get("action_type") in ("APPROVE","REJECT","COUNTER","REQUEST_INFO"):
                return d
    except Exception:
        pass
    return _rule_based(obs)

# ── Episode runner ─────────────────────────────────────────────────────────

def run_task(task_id, env_client, llm_client, seed=42):
    rewards=[]; steps_taken=0; score=0.0; success=False
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        reset_data=env_client.reset(task_id=task_id, seed=seed)
        obs_data=reset_data.get("observation") or reset_data
        done=False
        for step_n in range(1, MAX_STEPS+1):
            if done: break
            try:
                if llm_client:
                    action=_llm_action(llm_client, obs_data)
                else:
                    action=_rule_based(obs_data)
            except Exception as e:
                action=_rule_based(obs_data)
            action_str=(f"{action.get('action_type','?')}"
                        f"({str(action.get('reasoning',''))[:35]})")
            step_error=None
            try:
                sd=env_client.step(task_id=task_id, action=action)
                reward=float(sd.get("reward",0.0)); done=bool(sd.get("done",False))
                step_error=sd.get("error"); next_obs=sd.get("observation")
            except Exception as e:
                reward=0.0; done=True; step_error=str(e)[:60]; next_obs=None
            rewards.append(reward); steps_taken=step_n
            log_step(step=step_n, action=action_str, reward=reward, done=done, error=step_error)
            if done or next_obs is None: break
            obs_data=next_obs
        try:
            gd=env_client.grade(task_id=task_id)
            score=float(gd.get("final_score", gd.get("scores",{}).get("score",0.0)))
        except Exception:
            score=min(max(sum(rewards)/max(steps_taken*0.5,1),0.0),1.0)
        score=min(max(float(score),0.0),1.0); success=score>=SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[ERROR] run_task({task_id}): {e}", flush=True)
        score=0.0; success=False
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    env_client=CreditLensClient(API_BASE_URL)
    print(f"[INFO] Waiting for API at {API_BASE_URL} ...", flush=True)
    ready=False
    for attempt in range(30):
        if env_client.health():
            print(f"[INFO] API ready (attempt {attempt+1})", flush=True)
            ready=True; break
        time.sleep(2)
    if not ready:
        print("[ERROR] API not ready after 60s", flush=True)
        for t in ["easy","medium","hard"]:
            log_start(task=t, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0

    try:
        llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[WARN] LLM client init failed: {e}", flush=True)
        llm_client = None
    scores={}
    for task_id in ["easy","medium","hard"]:
        try:
            scores[task_id]=run_task(task_id, env_client, llm_client, seed=42)
        except Exception as e:
            print(f"[ERROR] outer({task_id}): {e}", flush=True)
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            scores[task_id]=0.0
    overall=sum(scores.values())/max(len(scores),1)
    print(f"\n[SUMMARY] easy={scores.get('easy',0):.3f} medium={scores.get('medium',0):.3f} "
          f"hard={scores.get('hard',0):.3f} overall={overall:.3f}", flush=True)

    

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] inference crash: {e}", flush=True)
        sys.exit(0)