import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    reward = 0.0
    if data_source == "sys_query":
        if "</think>" in solution_str:
            reward += 0.5
            pattern = r"^.*?\n</think>\n(.*$)"
            matches = re.match(pattern, solution_str, flags=re.DOTALL)
            if matches:
                reward += 0.5
                match = matches.group(1)
                if ground_truth.lower().strip() in match.lower():
                    reward += 1
    return reward


def no_think(data_source, solution_str, ground_truth, extra_info=None):
    reward = 0.0
    if ground_truth.lower().strip() in solution_str.lower():
        reward += 1
    return reward
