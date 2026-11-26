"""Pure selection policy; free KV pages are observations, not GPU utilization."""


class ReplicaSelector:
    def __init__(self, count, policy="most_free_kv"):
        if count < 1 or policy not in ("most_free_kv", "round_robin"):
            raise ValueError("Invalid replica count or routing policy")
        self.count, self.policy, self.cursor = count, policy, 0

    def choose(self, states=None):
        if self.policy == "round_robin":
            selected = self.cursor
        else:
            if states is None or len(states) != self.count:
                raise ValueError("Need a KV snapshot from every replica")
            for state in states:
                free, total = state.get("free_kv_blocks"), state.get("total_kv_blocks")
                if (not state.get("healthy") or type(free) is not int or type(total) is not int
                        or not 0 <= free <= total or total < 1):
                    raise ValueError("Invalid/unhealthy replica KV snapshot")
            signatures = {(s["model"], s["block_size"], s["max_model_len"], s["attention_backend"])
                          for s in states}
            if len(signatures) != 1:
                raise ValueError("Replicas must use the same model, page size, context and backend")
            # Cyclic tie breaking avoids permanently preferring GPU 0 when idle.
            selected = max(range(self.count), key=lambda i:
                           (states[i]["free_kv_blocks"], -((i-self.cursor) % self.count)))
        self.cursor = (selected + 1) % self.count
        return selected
