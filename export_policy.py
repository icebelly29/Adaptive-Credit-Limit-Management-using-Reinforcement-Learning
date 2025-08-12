from __future__ import annotations
import argparse, torch
from stable_baselines3 import PPO

# Export TorchScript that returns ONLY the action tensor for Java serving
class PolicyWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy = policy
    def forward(self, obs: torch.Tensor):
        # obs: [N, 6]
        dist = self.policy.get_distribution(obs)
        logits = dist.distribution.logits
        action = torch.argmax(logits, dim=1)
        return action  # [N]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="outputs/models/ppo_credit_limit.zip")
    p.add_argument("--out", default="outputs/models/model.pt")
    args = p.parse_args()

    model = PPO.load(args.model)
    wrapped = PolicyWrapper(model.policy).eval()
    example = torch.zeros(1, 6, dtype=torch.float32)
    traced = torch.jit.trace(wrapped, example)
    traced.save(args.out)
    print("Saved TorchScript to", args.out)

if __name__ == "__main__":
    main()
