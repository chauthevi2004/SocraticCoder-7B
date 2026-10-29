"""
Quick API Key Validator for SocraticCoder-7B.
Checks whether API keys in .env are valid and prints their status without leaking the secret.
"""
import os
from pathlib import Path

def load_env_file():
    env_paths = [
        Path(".env"),
        Path(__file__).resolve().parents[1] / ".env",
        Path("/home/aioz-vi/Downloads/.env")
    ]
    for p in env_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if v and k not in os.environ:
                            os.environ[k] = v

def mask_key(k: str) -> str:
    if len(k) <= 8:
        return "****"
    return k[:4] + "..." + k[-4:]

def main():
    load_env_file()
    print("=" * 60)
    print("SOCRATICCODER-7B: API KEY STATUS CHECK")
    print("=" * 60)
    
    keys = {
        "ANTHROPIC_API_KEY (Claude 3.5 Sonnet)": "ANTHROPIC_API_KEY",
        "MOONSHOT_API_KEY (Kimi / Moonshot)": "MOONSHOT_API_KEY",
        "GEMINI_API_KEY (Google Gemini)": "GEMINI_API_KEY",
        "OPENAI_API_KEY (GPT-4o / mini)": "OPENAI_API_KEY",
    }
    
    found_any = False
    for label, env_var in keys.items():
        val = os.getenv(env_var, "").strip()
        if val:
            found_any = True
            print(f"[FOUND] {label}: Configured [Key: {mask_key(val)}]")
        else:
            print(f"[EMPTY] {label}: Not configured")
            
    print("=" * 60)
    if not found_any:
        print("[INFO] No API keys configured in .env.")
        print("[INFO] Copy .env.example to .env and insert your keys.")
    else:
        print("[INFO] Ready for external API evaluations.")

if __name__ == "__main__":
    main()
