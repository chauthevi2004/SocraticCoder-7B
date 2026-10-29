"""
Verified Seed Bank of Paired Isomorphic Coding Problems (P, P') with Buggy Student
Submissions (c_0), Deterministic Unit Tests U(P) and U(P'), and Gold Socratic Hints.

Each pair (P, P') shares an identical underlying algorithmic invariant so that
conceptual Socratic guidance on P transfers zero-shot to P', whereas localized
line-editing hints fail to transfer.
"""

from __future__ import annotations

from typing import Any, Dict, List


SEED_PAIRED_PROBLEMS: List[Dict[str, Any]] = [
    # 1. Dynamic Programming: Unbounded 1D Minimization (Coin Change -> Rod Cutting / Min Squares)
    {
        "family_id": "dp_unbounded_min",
        "algorithmic_concept": "1D Unbounded Dynamic Programming (inclusive target boundary & unachievable sentinel check)",
        "misconception_type": "off_by_one_dp_table_size_and_range",
        "p_title": "Minimum Coins for Exact Change",
        "p_entry_point": "min_coins",
        "p_prompt": (
            "Write a function `min_coins(coins: list[int], amount: int) -> int` that returns "
            "the fewest number of coins needed to make up `amount`. If the amount cannot be "
            "made up by any combination of the coins, return -1."
        ),
        "p_buggy_code": (
            "def min_coins(coins: list[int], amount: int) -> int:\n"
            "    dp = [float('inf')] * amount\n"
            "    dp[0] = 0\n"
            "    for x in range(1, amount):\n"
            "        for c in coins:\n"
            "            if x - c >= 0:\n"
            "                dp[x] = min(dp[x], dp[x - c] + 1)\n"
            "    return dp[amount - 1] if dp[amount - 1] != float('inf') else -1\n"
        ),
        "p_canonical_solution": (
            "def min_coins(coins: list[int], amount: int) -> int:\n"
            "    dp = [float('inf')] * (amount + 1)\n"
            "    dp[0] = 0\n"
            "    for x in range(1, amount + 1):\n"
            "        for c in coins:\n"
            "            if x - c >= 0:\n"
            "                dp[x] = min(dp[x], dp[x - c] + 1)\n"
            "    return dp[amount] if dp[amount] != float('inf') else -1\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([1, 2, 5], 11) == 3\n"
            "    assert candidate([2], 3) == -1\n"
            "    assert candidate([1], 0) == 0\n"
            "    assert candidate([3, 5], 9) == 3\n"
            "    assert candidate([4, 5], 7) == -1\n"
        ),
        "p_prime_title": "Minimum Packages for Exact Weight",
        "p_prime_entry_point": "min_packages",
        "p_prime_prompt": (
            "Write a function `min_packages(weights: list[int], target: int) -> int` that returns "
            "the minimum number of packages (unlimited supply of each weight) required to reach "
            "an exact total weight of `target`. Return -1 if `target` cannot be formed."
        ),
        "p_prime_starter_code": (
            "def min_packages(weights: list[int], target: int) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def min_packages(weights: list[int], target: int) -> int:\n"
            "    dp = [float('inf')] * (target + 1)\n"
            "    dp[0] = 0\n"
            "    for w_sum in range(1, target + 1):\n"
            "        for w in weights:\n"
            "            if w_sum - w >= 0:\n"
            "                dp[w_sum] = min(dp[w_sum], dp[w_sum - w] + 1)\n"
            "    return dp[target] if dp[target] != float('inf') else -1\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 3, 7], 14) == 2\n"
            "    assert candidate([4, 6], 5) == -1\n"
            "    assert candidate([5], 0) == 0\n"
            "    assert candidate([3, 4], 10) == 3\n"
            "    assert candidate([2, 5], 11) == 4\n"
        ),
        "gold_socratic_hint": (
            "Think about what index in your DP table represents the exact target value `N` when "
            "index `0` represents a sum of zero. If you need indices from `0` up to `N` inclusive, "
            "how many elements must the table allocate, and what happens when `N == 0`?"
        ),
    },

    # 2. Binary Search on Monotonic Predicate (Koko Bananas -> Ship Packages Within D Days)
    {
        "family_id": "binary_search_monotonic",
        "algorithmic_concept": "Binary Search on Answer Space (half-open/closed loop termination & ceiling division)",
        "misconception_type": "infinite_loop_mid_update_and_float_truncation",
        "p_title": "Minimum Eating Speed",
        "p_entry_point": "min_eating_speed",
        "p_prompt": (
            "Write a function `min_eating_speed(piles: list[int], h: int) -> int` that returns "
            "the minimum integer rate `k` such that all piles can be finished within `h` hours, "
            "where a pile of size `p` takes `ceil(p / k)` hours."
        ),
        "p_buggy_code": (
            "def min_eating_speed(piles: list[int], h: int) -> int:\n"
            "    lo, hi = 1, max(piles)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        hours = sum(p // mid for p in piles)\n"
            "        if hours <= h:\n"
            "            hi = mid - 1\n"
            "        else:\n"
            "            lo = mid + 1\n"
            "    return lo\n"
        ),
        "p_canonical_solution": (
            "def min_eating_speed(piles: list[int], h: int) -> int:\n"
            "    lo, hi = 1, max(piles)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        hours = sum((p + mid - 1) // mid for p in piles)\n"
            "        if hours <= h:\n"
            "            hi = mid\n"
            "        else:\n"
            "            lo = mid + 1\n"
            "    return lo\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([3, 6, 7, 11], 8) == 4\n"
            "    assert candidate([30, 11, 23, 4, 20], 5) == 30\n"
            "    assert candidate([30, 11, 23, 4, 20], 6) == 23\n"
            "    assert candidate([10], 4) == 3\n"
        ),
        "p_prime_title": "Minimum Printer Throughput",
        "p_prime_entry_point": "min_printer_rate",
        "p_prime_prompt": (
            "Write a function `min_printer_rate(jobs: list[int], max_hours: int) -> int` that returns "
            "the smallest integer pages-per-hour rate `r` needed to finish all print `jobs` within "
            "`max_hours`, where a job of `j` pages takes `ceil(j / r)` hours."
        ),
        "p_prime_starter_code": (
            "def min_printer_rate(jobs: list[int], max_hours: int) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def min_printer_rate(jobs: list[int], max_hours: int) -> int:\n"
            "    lo, hi = 1, max(jobs)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        hours = sum((j + mid - 1) // mid for j in jobs)\n"
            "        if hours <= max_hours:\n"
            "            hi = mid\n"
            "        else:\n"
            "            lo = mid + 1\n"
            "    return lo\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([5, 8, 12], 6) == 5\n"
            "    assert candidate([15, 10, 20], 3) == 20\n"
            "    assert candidate([9], 2) == 5\n"
            "    assert candidate([4, 9, 11, 16], 8) == 6\n"
        ),
        "gold_socratic_hint": (
            "Check two invariants in your binary search: first, how does integer floor division `//` "
            "behave when there are leftover items in a batch instead of rounding up? Second, if `mid` "
            "is a valid feasible rate, could `mid` itself be the minimum valid answer, and what happens "
            "if you discard it by setting the upper bound to `mid - 1`?"
        ),
    },

    # 3. Sliding Window Variable Length (Longest Subarray Sum <= K -> Longest Substring At Most K Distinct)
    {
        "family_id": "sliding_window_contraction",
        "algorithmic_concept": "Two-Pointer Sliding Window Contraction Invariant (while loop vs single if & inclusive window length)",
        "misconception_type": "if_instead_of_while_and_off_by_one_length",
        "p_title": "Longest Subarray Within Budget",
        "p_entry_point": "longest_within_budget",
        "p_prompt": (
            "Write a function `longest_within_budget(costs: list[int], budget: int) -> int` that "
            "returns the maximum length of a contiguous subarray of non-negative `costs` whose "
            "sum is less than or equal to `budget`."
        ),
        "p_buggy_code": (
            "def longest_within_budget(costs: list[int], budget: int) -> int:\n"
            "    left = 0\n"
            "    curr_sum = 0\n"
            "    best = 0\n"
            "    for right in range(len(costs)):\n"
            "        curr_sum += costs[right]\n"
            "        if curr_sum > budget:\n"
            "            curr_sum -= costs[left]\n"
            "            left += 1\n"
            "        best = max(best, right - left)\n"
            "    return best\n"
        ),
        "p_canonical_solution": (
            "def longest_within_budget(costs: list[int], budget: int) -> int:\n"
            "    left = 0\n"
            "    curr_sum = 0\n"
            "    best = 0\n"
            "    for right in range(len(costs)):\n"
            "        curr_sum += costs[right]\n"
            "        while curr_sum > budget and left <= right:\n"
            "            curr_sum -= costs[left]\n"
            "            left += 1\n"
            "        best = max(best, right - left + 1)\n"
            "    return best\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([1, 1, 1, 1, 10, 1], 4) == 4\n"
            "    assert candidate([5, 5, 5], 4) == 0\n"
            "    assert candidate([2, 1, 1, 3], 4) == 3\n"
            "    assert candidate([3], 3) == 1\n"
        ),
        "p_prime_title": "Longest Contiguous Segment Under Bandwidth Cap",
        "p_prime_entry_point": "max_segment_len",
        "p_prime_prompt": (
            "Write a function `max_segment_len(packets: list[int], cap: int) -> int` that returns "
            "the length of the longest contiguous subarray of non-negative `packets` whose total "
            "size does not exceed `cap`."
        ),
        "p_prime_starter_code": (
            "def max_segment_len(packets: list[int], cap: int) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def max_segment_len(packets: list[int], cap: int) -> int:\n"
            "    left = 0\n"
            "    total = 0\n"
            "    ans = 0\n"
            "    for right in range(len(packets)):\n"
            "        total += packets[right]\n"
            "        while total > cap and left <= right:\n"
            "            total -= packets[left]\n"
            "            left += 1\n"
            "        ans = max(ans, right - left + 1)\n"
            "    return ans\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 2, 2, 20, 1, 1], 6) == 3\n"
            "    assert candidate([10, 12], 5) == 0\n"
            "    assert candidate([1, 2, 3, 1, 1], 5) == 3\n"
            "    assert candidate([4], 5) == 1\n"
        ),
        "gold_socratic_hint": (
            "Trace what happens when a single large element causes the window sum to exceed the limit "
            "by more than the value at `left`—does shrinking `left` just once restore a valid window? "
            "Also, if `left == right` on a valid single-element window, what should its length be?"
        ),
    },

    # 4. Prefix Sum + Hash Map Frequency (Subarray Sum Equals K -> Zero-Modulo / Target Delta Subarrays)
    {
        "family_id": "prefix_sum_hashmap",
        "algorithmic_concept": "Prefix Sum Frequency Hash Map (initializing empty prefix 0:1 and updating count after query)",
        "misconception_type": "missing_base_prefix_zero_and_update_order",
        "p_title": "Count Subarrays With Target Sum",
        "p_entry_point": "subarray_sum_count",
        "p_prompt": (
            "Write a function `subarray_sum_count(nums: list[int], k: int) -> int` that returns "
            "the total number of contiguous subarrays whose sum equals `k`."
        ),
        "p_buggy_code": (
            "def subarray_sum_count(nums: list[int], k: int) -> int:\n"
            "    seen = {}\n"
            "    prefix = 0\n"
            "    count = 0\n"
            "    for x in nums:\n"
            "        prefix += x\n"
            "        seen[prefix] = seen.get(prefix, 0) + 1\n"
            "        if prefix - k in seen:\n"
            "            count += seen[prefix - k]\n"
            "    return count\n"
        ),
        "p_canonical_solution": (
            "def subarray_sum_count(nums: list[int], k: int) -> int:\n"
            "    seen = {0: 1}\n"
            "    prefix = 0\n"
            "    count = 0\n"
            "    for x in nums:\n"
            "        prefix += x\n"
            "        if prefix - k in seen:\n"
            "            count += seen[prefix - k]\n"
            "        seen[prefix] = seen.get(prefix, 0) + 1\n"
            "    return count\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([1, 1, 1], 2) == 2\n"
            "    assert candidate([1, 2, 3], 3) == 2\n"
            "    assert candidate([1, -1, 0], 0) == 3\n"
            "    assert candidate([3], 0) == 0\n"
        ),
        "p_prime_title": "Count Intervals With Exact Net Profit",
        "p_prime_entry_point": "count_profit_intervals",
        "p_prime_prompt": (
            "Write a function `count_profit_intervals(deltas: list[int], target: int) -> int` "
            "that returns the number of contiguous subarrays in `deltas` whose sum equals `target`."
        ),
        "p_prime_starter_code": (
            "def count_profit_intervals(deltas: list[int], target: int) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def count_profit_intervals(deltas: list[int], target: int) -> int:\n"
            "    seen = {0: 1}\n"
            "    curr = 0\n"
            "    total = 0\n"
            "    for d in deltas:\n"
            "        curr += d\n"
            "        total += seen.get(curr - target, 0)\n"
            "        seen[curr] = seen.get(curr, 0) + 1\n"
            "    return total\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 2, 2], 4) == 2\n"
            "    assert candidate([4, -4, 4], 0) == 2\n"
            "    assert candidate([5], 0) == 0\n"
            "    assert candidate([1, 2, 1, 2, 1], 3) == 4\n"
        ),
        "gold_socratic_hint": (
            "Consider two edge cases in your prefix map: first, how do you account for a subarray "
            "that starts at index 0 where `prefix == k` (so `prefix - k == 0`)? Second, if `k == 0` "
            "and you insert `prefix` into `seen` before querying `prefix - k`, what unintended "
            "empty subarray are you counting?"
        ),
    },

    # 5. Monotonic Stack (Daily Temperatures -> Next Greater Price Jump)
    {
        "family_id": "monotonic_stack_next_greater",
        "algorithmic_concept": "Decreasing Monotonic Stack storing indices to compute distance to next strictly greater element",
        "misconception_type": "storing_values_instead_of_indices_and_weak_inequality",
        "p_title": "Days Until Warmer Temperature",
        "p_entry_point": "warmer_days",
        "p_prompt": (
            "Write a function `warmer_days(temps: list[int]) -> list[int]` that returns an array "
            "`ans` where `ans[i]` is the number of days you have to wait after day `i` to get a "
            "strictly warmer temperature, or 0 if no warmer day exists."
        ),
        "p_buggy_code": (
            "def warmer_days(temps: list[int]) -> list[int]:\n"
            "    ans = [0] * len(temps)\n"
            "    stack = []\n"
            "    for i, t in enumerate(temps):\n"
            "        while stack and t >= temps[stack[-1]]:\n"
            "            prev = stack.pop()\n"
            "            ans[prev] = t - temps[prev]\n"
            "        stack.append(i)\n"
            "    return ans\n"
        ),
        "p_canonical_solution": (
            "def warmer_days(temps: list[int]) -> list[int]:\n"
            "    ans = [0] * len(temps)\n"
            "    stack = []\n"
            "    for i, t in enumerate(temps):\n"
            "        while stack and t > temps[stack[-1]]:\n"
            "            prev = stack.pop()\n"
            "            ans[prev] = i - prev\n"
            "        stack.append(i)\n"
            "    return ans\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]\n"
            "    assert candidate([30, 30, 30]) == [0, 0, 0]\n"
            "    assert candidate([30, 40, 50, 60]) == [1, 1, 1, 0]\n"
        ),
        "p_prime_title": "Steps Until Higher Stock Price",
        "p_prime_entry_point": "days_until_higher_price",
        "p_prime_prompt": (
            "Write a function `days_until_higher_price(prices: list[int]) -> list[int]` that "
            "returns for each index `i` the number of days until a strictly higher stock price "
            "occurs, or 0 if no future day has a strictly higher price."
        ),
        "p_prime_starter_code": (
            "def days_until_higher_price(prices: list[int]) -> list[int]:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def days_until_higher_price(prices: list[int]) -> list[int]:\n"
            "    res = [0] * len(prices)\n"
            "    stack = []\n"
            "    for i, p in enumerate(prices):\n"
            "        while stack and p > prices[stack[-1]]:\n"
            "            idx = stack.pop()\n"
            "            res[idx] = i - idx\n"
            "        stack.append(i)\n"
            "    return res\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([10, 10, 12, 9, 15]) == [2, 1, 2, 1, 0]\n"
            "    assert candidate([50, 40, 30]) == [0, 0, 0]\n"
            "    assert candidate([5, 6, 7]) == [1, 1, 0]\n"
        ),
        "gold_socratic_hint": (
            "Look closely at what the problem asks you to store in `ans[prev]`: is it the difference "
            "in values or the distance in index positions? Also, if two days have the exact same "
            "value, should an equal value trigger the strictly greater condition?"
        ),
    },

    # 6. Interval Greedy Scheduling (Non-overlapping Intervals -> Maximum Compatible Meetings)
    {
        "family_id": "interval_greedy_end_sort",
        "algorithmic_concept": "Greedy Interval Scheduling by sorting on finish time (x[1]) rather than start time (x[0])",
        "misconception_type": "sorting_by_start_time_instead_of_end_time",
        "p_title": "Maximum Non-Overlapping Intervals",
        "p_entry_point": "max_non_overlapping",
        "p_prompt": (
            "Write a function `max_non_overlapping(intervals: list[list[int]]) -> int` that "
            "returns the maximum number of mutually non-overlapping intervals you can select "
            "(intervals touching at endpoints like [1,2] and [2,3] do not overlap)."
        ),
        "p_buggy_code": (
            "def max_non_overlapping(intervals: list[list[int]]) -> int:\n"
            "    if not intervals:\n"
            "        return 0\n"
            "    intervals.sort(key=lambda x: x[0])\n"
            "    count = 1\n"
            "    last_end = intervals[0][1]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start >= last_end:\n"
            "            count += 1\n"
            "            last_end = end\n"
            "    return count\n"
        ),
        "p_canonical_solution": (
            "def max_non_overlapping(intervals: list[list[int]]) -> int:\n"
            "    if not intervals:\n"
            "        return 0\n"
            "    intervals.sort(key=lambda x: x[1])\n"
            "    count = 1\n"
            "    last_end = intervals[0][1]\n"
            "    for start, end in intervals[1:]:\n"
            "        if start >= last_end:\n"
            "            count += 1\n"
            "            last_end = end\n"
            "    return count\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([[1, 10], [2, 3], [3, 4], [4, 5]]) == 3\n"
            "    assert candidate([[1, 2], [1, 2], [1, 2]]) == 1\n"
            "    assert candidate([[1, 2], [2, 3]]) == 2\n"
            "    assert candidate([]) == 0\n"
        ),
        "p_prime_title": "Maximum Scheduled Classroom Lectures",
        "p_prime_entry_point": "max_lectures",
        "p_prime_prompt": (
            "Write a function `max_lectures(classes: list[list[int]]) -> int` that returns the "
            "maximum number of classes `[start, end]` that can be scheduled in a single room "
            "without overlap (where `next_start >= prev_end` is allowed)."
        ),
        "p_prime_starter_code": (
            "def max_lectures(classes: list[list[int]]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def max_lectures(classes: list[list[int]]) -> int:\n"
            "    if not classes:\n"
            "        return 0\n"
            "    classes.sort(key=lambda x: x[1])\n"
            "    taken = 1\n"
            "    end_time = classes[0][1]\n"
            "    for s, e in classes[1:]:\n"
            "        if s >= end_time:\n"
            "            taken += 1\n"
            "            end_time = e\n"
            "    return taken\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([[0, 20], [1, 5], [5, 10], [10, 15]]) == 3\n"
            "    assert candidate([[2, 4], [2, 4]]) == 1\n"
            "    assert candidate([[1, 3], [3, 6], [6, 8]]) == 3\n"
            "    assert candidate([]) == 0\n"
        ),
        "gold_socratic_hint": (
            "Imagine the first interval starts very early at `t = 1` but ends very late at `t = 100`, "
            "while three short intervals `[2, 3], [3, 4], [4, 5]` sit inside it. Which interval attribute "
            "(start time or finish time) should you greedily minimize to leave the most room for future intervals?"
        ),
    },

    # 7. House Robber / Non-Adjacent 1D DP (Max Non-Adjacent Sum -> Max Non-Adjacent Tasks)
    {
        "family_id": "dp_non_adjacent",
        "algorithmic_concept": "1D State Transition for Non-Adjacent Selection: dp[i] = max(dp[i-1], dp[i-2] + val[i])",
        "misconception_type": "even_odd_index_partition_fallacy",
        "p_title": "Maximum Non-Adjacent Vault Sum",
        "p_entry_point": "rob_vaults",
        "p_prompt": (
            "Write a function `rob_vaults(nums: list[int]) -> int` that returns the maximum "
            "sum of elements you can pick from `nums` such that no two picked elements are adjacent."
        ),
        "p_buggy_code": (
            "def rob_vaults(nums: list[int]) -> int:\n"
            "    if not nums:\n"
            "        return 0\n"
            "    return max(sum(nums[0::2]), sum(nums[1::2]))\n"
        ),
        "p_canonical_solution": (
            "def rob_vaults(nums: list[int]) -> int:\n"
            "    prev2, prev1 = 0, 0\n"
            "    for x in nums:\n"
            "        prev2, prev1 = prev1, max(prev1, prev2 + x)\n"
            "    return prev1\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 1, 1, 2]) == 4\n"
            "    assert candidate([1, 2, 3, 1]) == 4\n"
            "    assert candidate([2, 7, 9, 3, 1]) == 12\n"
            "    assert candidate([]) == 0\n"
        ),
        "p_prime_title": "Maximum Non-Consecutive Energy Boost",
        "p_prime_entry_point": "max_energy_boost",
        "p_prime_prompt": (
            "Write a function `max_energy_boost(boosts: list[int]) -> int` that returns the "
            "maximum total energy obtainable from `boosts` without selecting any two consecutive items."
        ),
        "p_prime_starter_code": (
            "def max_energy_boost(boosts: list[int]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def max_energy_boost(boosts: list[int]) -> int:\n"
            "    a, b = 0, 0\n"
            "    for v in boosts:\n"
            "        a, b = b, max(b, a + v)\n"
            "    return b\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([5, 1, 1, 5]) == 10\n"
            "    assert candidate([3, 10, 3, 1, 2]) == 12\n"
            "    assert candidate([4]) == 4\n"
            "    assert candidate([]) == 0\n"
        ),
        "gold_socratic_hint": (
            "Trace your logic on the array `[2, 1, 1, 2]`: does an optimal choice always stay strictly "
            "on even indices or strictly on odd indices, or can it skip two elements in a row? At each "
            "step `i`, what are the two recurrence choices based on the best totals at `i - 1` and `i - 2`?"
        ),
    },

    # 8. Graph / Topological Cycle Detection (Course Schedule -> Task Dependency Deadlock)
    {
        "family_id": "graph_cycle_detection_dfs",
        "algorithmic_concept": "Directed Graph Cycle Detection via 3-State DFS Coloring (0=unvisited, 1=visiting on current path, 2=visited)",
        "misconception_type": "two_state_visited_conflates_cross_edge_with_back_edge",
        "p_title": "Can Finish All Courses",
        "p_entry_point": "can_finish",
        "p_prompt": (
            "Write a function `can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool` "
            "that returns `True` if it is possible to finish all courses given directed edges `[u, v]` "
            "(meaning `u` depends on `v`), or `False` if there is a cycle."
        ),
        "p_buggy_code": (
            "def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:\n"
            "    adj = {i: [] for i in range(num_courses)}\n"
            "    for u, v in prerequisites:\n"
            "        adj[u].append(v)\n"
            "    visited = set()\n"
            "    def dfs(node):\n"
            "        if node in visited:\n"
            "            return False\n"
            "        visited.add(node)\n"
            "        for nxt in adj[node]:\n"
            "            if not dfs(nxt):\n"
            "                return False\n"
            "        return True\n"
            "    return all(dfs(i) for i in range(num_courses))\n"
        ),
        "p_canonical_solution": (
            "def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:\n"
            "    adj = {i: [] for i in range(num_courses)}\n"
            "    for u, v in prerequisites:\n"
            "        adj[u].append(v)\n"
            "    state = [0] * num_courses\n"
            "    def dfs(node):\n"
            "        if state[node] == 1:\n"
            "            return False\n"
            "        if state[node] == 2:\n"
            "            return True\n"
            "        state[node] = 1\n"
            "        for nxt in adj[node]:\n"
            "            if not dfs(nxt):\n"
                "                return False\n"
            "        state[node] = 2\n"
            "        return True\n"
            "    return all(dfs(i) for i in range(num_courses))\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate(2, [[1, 0]]) is True\n"
            "    assert candidate(2, [[1, 0], [0, 1]]) is False\n"
            "    assert candidate(3, [[0, 1], [0, 2], [1, 2]]) is True\n"
            "    assert candidate(4, [[0, 1], [1, 2], [2, 3], [3, 1]]) is False\n"
        ),
        "p_prime_title": "Detect Build Pipeline Deadlock",
        "p_prime_entry_point": "is_valid_pipeline",
        "p_prime_prompt": (
            "Write a function `is_valid_pipeline(n_stages: int, deps: list[list[int]]) -> bool` "
            "that returns `True` if a directed build graph with `n_stages` (`0` to `n_stages - 1`) "
            "and dependencies `[a, b]` has no circular dependencies, and `False` otherwise."
        ),
        "p_prime_starter_code": (
            "def is_valid_pipeline(n_stages: int, deps: list[list[int]]) -> bool:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def is_valid_pipeline(n_stages: int, deps: list[list[int]]) -> bool:\n"
            "    adj = {i: [] for i in range(n_stages)}\n"
            "    for a, b in deps:\n"
            "        adj[a].append(b)\n"
            "    color = [0] * n_stages\n"
            "    def dfs(u):\n"
            "        if color[u] == 1:\n"
            "            return False\n"
            "        if color[u] == 2:\n"
            "            return True\n"
            "        color[u] = 1\n"
            "        for v in adj[u]:\n"
            "            if not dfs(v):\n"
            "                return False\n"
            "        color[u] = 2\n"
            "        return True\n"
            "    return all(dfs(i) for i in range(n_stages))\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate(3, [[0, 1], [0, 2], [1, 2]]) is True\n"
            "    assert candidate(3, [[0, 1], [1, 2], [2, 0]]) is False\n"
            "    assert candidate(4, [[0, 1], [0, 2], [1, 3], [2, 3]]) is True\n"
        ),
        "gold_socratic_hint": (
            "Trace your `visited` set on a diamond-shaped DAG like `0 -> 1, 0 -> 2, 1 -> 2` (which has "
            "no cycle): when node `2` is reached a second time from `0` after `1 -> 2` already finished, "
            "is node `2` still on the active recursion stack or is it already completely explored? How can "
            "you distinguish between 'currently visiting on the active path' and 'already verified safe'?"
        ),
    },

    # 9. Kadane's Maximum Subarray (All-Negative Handling)
    {
        "family_id": "kadane_all_negative",
        "algorithmic_concept": "Kadane's Dynamic Programming Invariant for non-empty contiguous subarray (initializing with nums[0])",
        "misconception_type": "zero_initialization_fails_on_all_negative_arrays",
        "p_title": "Maximum Non-Empty Subarray Sum",
        "p_entry_point": "max_subarray_sum",
        "p_prompt": (
            "Write a function `max_subarray_sum(nums: list[int]) -> int` that finds the non-empty "
            "contiguous subarray within `nums` that has the largest sum and returns its sum."
        ),
        "p_buggy_code": (
            "def max_subarray_sum(nums: list[int]) -> int:\n"
            "    best = 0\n"
            "    curr = 0\n"
            "    for x in nums:\n"
            "        curr = max(0, curr + x)\n"
            "        best = max(best, curr)\n"
            "    return best\n"
        ),
        "p_canonical_solution": (
            "def max_subarray_sum(nums: list[int]) -> int:\n"
            "    best = nums[0]\n"
            "    curr = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        curr = max(x, curr + x)\n"
            "        best = max(best, curr)\n"
            "    return best\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n"
            "    assert candidate([-5, -2, -9]) == -2\n"
            "    assert candidate([-1]) == -1\n"
            "    assert candidate([5, 4, -1, 7, 8]) == 23\n"
        ),
        "p_prime_title": "Best Contiguous Signal Gain",
        "p_prime_entry_point": "best_signal_gain",
        "p_prime_prompt": (
            "Write a function `best_signal_gain(gains: list[int]) -> int` that returns the "
            "maximum sum of any non-empty contiguous slice of `gains` (which may contain all negative values)."
        ),
        "p_prime_starter_code": (
            "def best_signal_gain(gains: list[int]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def best_signal_gain(gains: list[int]) -> int:\n"
            "    best = gains[0]\n"
            "    curr = gains[0]\n"
            "    for g in gains[1:]:\n"
            "        curr = max(g, curr + g)\n"
            "        best = max(best, curr)\n"
            "    return best\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([-10, -3, -7]) == -3\n"
            "    assert candidate([3, -2, 5, -1]) == 6\n"
            "    assert candidate([-4]) == -4\n"
            "    assert candidate([1, 2, 3]) == 6\n"
        ),
        "gold_socratic_hint": (
            "What does your function return when every number in the input array is negative, such as "
            "`[-5, -2, -9]`? Since the subarray must be non-empty, what should `curr` and `best` be "
            "initialized to instead of `0`, and should a fresh subarray start at `x` or `0`?"
        ),
    },

    # 10. 2D Grid DP Path Minimum (Minimum Path Sum -> Minimum Cost Grid Traversal)
    {
        "family_id": "grid_dp_boundary_init",
        "algorithmic_concept": "2D Grid Dynamic Programming first-row and first-column cumulative prefix initialization",
        "misconception_type": "uninitialized_first_row_and_col_transitions",
        "p_title": "Minimum Grid Path Sum",
        "p_entry_point": "min_path_sum",
        "p_prompt": (
            "Write a function `min_path_sum(grid: list[list[int]]) -> int` that returns the minimum "
            "sum of numbers along a path from top-left `(0, 0)` to bottom-right `(m-1, n-1)`, moving "
            "only right or down."
        ),
        "p_buggy_code": (
            "def min_path_sum(grid: list[list[int]]) -> int:\n"
            "    m, n = len(grid), len(grid[0])\n"
            "    dp = [[0] * n for _ in range(m)]\n"
            "    dp[0][0] = grid[0][0]\n"
            "    for r in range(1, m):\n"
            "        for c in range(1, n):\n"
            "            dp[r][c] = grid[r][c] + min(dp[r - 1][c], dp[r][c - 1])\n"
            "    return dp[m - 1][n - 1]\n"
        ),
        "p_canonical_solution": (
            "def min_path_sum(grid: list[list[int]]) -> int:\n"
            "    m, n = len(grid), len(grid[0])\n"
            "    dp = [[0] * n for _ in range(m)]\n"
            "    dp[0][0] = grid[0][0]\n"
            "    for r in range(1, m):\n"
            "        dp[r][0] = dp[r - 1][0] + grid[r][0]\n"
            "    for c in range(1, n):\n"
            "        dp[0][c] = dp[0][c - 1] + grid[0][c]\n"
            "    for r in range(1, m):\n"
            "        for c in range(1, n):\n"
            "            dp[r][c] = grid[r][c] + min(dp[r - 1][c], dp[r][c - 1])\n"
            "    return dp[m - 1][n - 1]\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7\n"
            "    assert candidate([[1, 2, 3], [4, 5, 6]]) == 12\n"
            "    assert candidate([[5]]) == 5\n"
            "    assert candidate([[1, 2], [1, 1]]) == 3\n"
        ),
        "p_prime_title": "Minimum Toll Matrix Route",
        "p_prime_entry_point": "min_toll_route",
        "p_prime_prompt": (
            "Write a function `min_toll_route(tolls: list[list[int]]) -> int` that returns the "
            "minimum total toll to move from `(0, 0)` to `(rows-1, cols-1)` moving only right or down."
        ),
        "p_prime_starter_code": (
            "def min_toll_route(tolls: list[list[int]]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def min_toll_route(tolls: list[list[int]]) -> int:\n"
            "    m, n = len(tolls), len(tolls[0])\n"
            "    dp = [[0] * n for _ in range(m)]\n"
            "    dp[0][0] = tolls[0][0]\n"
            "    for r in range(1, m):\n"
            "        dp[r][0] = dp[r - 1][0] + tolls[r][0]\n"
            "    for c in range(1, n):\n"
            "        dp[0][c] = dp[0][c - 1] + tolls[0][c]\n"
            "    for r in range(1, m):\n"
            "        for c in range(1, n):\n"
            "            dp[r][c] = tolls[r][c] + min(dp[r - 1][c], dp[r][c - 1])\n"
            "    return dp[m - 1][n - 1]\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([[2, 4, 1], [3, 1, 2], [5, 2, 1]]) == 9\n"
            "    assert candidate([[3, 1, 2]]) == 6\n"
            "    assert candidate([[2], [4], [1]]) == 7\n"
        ),
        "gold_socratic_hint": (
            "Inspect the values in the first row `dp[0][c]` and the first column `dp[r][0]` before your "
            "nested loop starts at `r = 1, c = 1`: what values do those cells currently hold, and how "
            "does that affect `min(dp[r - 1][c], dp[r][c - 1])`?"
        ),
    },

    # 11. Two Pointers Opposite Ends (Container With Most Water -> Max Rectangular Trench Area)
    {
        "family_id": "two_pointers_opposite_ends",
        "algorithmic_concept": "Two-Pointer Inward Contraction by advancing the pointer with the smaller limiting height",
        "misconception_type": "moving_wrong_pointer_or_off_by_one_width",
        "p_title": "Maximum Water Container Area",
        "p_entry_point": "max_water_area",
        "p_prompt": (
            "Write a function `max_water_area(heights: list[int]) -> int` that finds two vertical lines "
            "at indices `i < j` maximizing `(j - i) * min(heights[i], heights[j])` and returns that maximum area."
        ),
        "p_buggy_code": (
            "def max_water_area(heights: list[int]) -> int:\n"
            "    left, right = 0, len(heights) - 1\n"
            "    best = 0\n"
            "    while left < right:\n"
            "        area = (right - left) * min(heights[left], heights[right])\n"
            "        best = max(best, area)\n"
            "        if heights[left] > heights[right]:\n"
            "            left += 1\n"
            "        else:\n"
            "            right -= 1\n"
            "    return best\n"
        ),
        "p_canonical_solution": (
            "def max_water_area(heights: list[int]) -> int:\n"
            "    left, right = 0, len(heights) - 1\n"
            "    best = 0\n"
            "    while left < right:\n"
            "        area = (right - left) * min(heights[left], heights[right])\n"
            "        best = max(best, area)\n"
            "        if heights[left] < heights[right]:\n"
            "            left += 1\n"
            "        else:\n"
            "            right -= 1\n"
            "    return best\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49\n"
            "    assert candidate([1, 1]) == 1\n"
            "    assert candidate([4, 3, 2, 1, 4]) == 16\n"
            "    assert candidate([1, 2, 1]) == 2\n"
        ),
        "p_prime_title": "Maximum Solar Panel Span Capacity",
        "p_prime_entry_point": "max_solar_span",
        "p_prime_prompt": (
            "Write a function `max_solar_span(pillars: list[int]) -> int` that returns the maximum "
            "rectangular span `(j - i) * min(pillars[i], pillars[j])` over all `0 <= i < j < len(pillars)`."
        ),
        "p_prime_starter_code": (
            "def max_solar_span(pillars: list[int]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def max_solar_span(pillars: list[int]) -> int:\n"
            "    l, r = 0, len(pillars) - 1\n"
            "    ans = 0\n"
            "    while l < r:\n"
            "        ans = max(ans, (r - l) * min(pillars[l], pillars[r]))\n"
            "        if pillars[l] < pillars[r]:\n"
            "            l += 1\n"
            "        else:\n"
            "            r -= 1\n"
            "    return ans\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 9, 5, 3, 8, 7]) == 28\n"
            "    assert candidate([5, 5]) == 5\n"
            "    assert candidate([6, 1, 6]) == 12\n"
        ),
        "gold_socratic_hint": (
            "Since shrinking the width `right - left` always decreases the horizontal distance, the only way "
            "to find a larger area is to increase the limiting height `min(heights[left], heights[right])`. "
            "Which pointer—the shorter wall or the taller wall—currently caps the height and must be moved inward?"
        ),
    },

    # 12. Shortest Path Dijkstra Priority Queue (Network Delay Time -> Minimum Latency Routing)
    {
        "family_id": "dijkstra_shortest_path_heap",
        "algorithmic_concept": "Dijkstra's Shortest Path with Min-Heap (skipping stale heap entries and taking max over all reachable nodes)",
        "misconception_type": " summing_distances_instead_of_max_or_missing_unreached_check",
        "p_title": "Network Signal Broadcast Delay",
        "p_entry_point": "network_delay",
        "p_prompt": (
            "Write a function `network_delay(times: list[list[int]], n: int, k: int) -> int` where "
            "`times[i] = [u, v, w]` is a directed edge from `u` to `v` (1-indexed `1..n`) with weight `w`. "
            "Return the minimum time for all `n` nodes to receive a signal starting from node `k`, or `-1` if impossible."
        ),
        "p_buggy_code": (
            "import heapq\n"
            "def network_delay(times: list[list[int]], n: int, k: int) -> int:\n"
            "    adj = {i: [] for i in range(1, n + 1)}\n"
            "    for u, v, w in times:\n"
            "        adj[u].append((v, w))\n"
            "    dist = {k: 0}\n"
            "    pq = [(0, k)]\n"
            "    while pq:\n"
            "        d, u = heapq.heappop(pq)\n"
            "        for v, w in adj[u]:\n"
            "            if v not in dist:\n"
            "                dist[v] = d + w\n"
            "                heapq.heappush(pq, (dist[v], v))\n"
            "    return max(dist.values()) if len(dist) == n else -1\n"
        ),
        "p_canonical_solution": (
            "import heapq\n"
            "def network_delay(times: list[list[int]], n: int, k: int) -> int:\n"
            "    adj = {i: [] for i in range(1, n + 1)}\n"
            "    for u, v, w in times:\n"
            "        adj[u].append((v, w))\n"
            "    dist = {k: 0}\n"
            "    pq = [(0, k)]\n"
            "    while pq:\n"
            "        d, u = heapq.heappop(pq)\n"
            "        if d > dist.get(u, float('inf')):\n"
            "            continue\n"
            "        for v, w in adj[u]:\n"
            "            if d + w < dist.get(v, float('inf')):\n"
            "                dist[v] = d + w\n"
            "                heapq.heappush(pq, (dist[v], v))\n"
            "    return max(dist.values()) if len(dist) == n else -1\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2\n"
            "    assert candidate([[1, 2, 10], [1, 3, 2], [3, 2, 3]], 3, 1) == 5\n"
            "    assert candidate([[1, 2, 1]], 2, 2) == -1\n"
        ),
        "p_prime_title": "Minimum Relay Propagation Time",
        "p_prime_entry_point": "relay_propagation_time",
        "p_prime_prompt": (
            "Write a function `relay_propagation_time(links: list[list[int]], num_nodes: int, source: int) -> int` "
            "that returns the time for a packet from `source` (1-indexed `1..num_nodes`) to reach every node "
            "along directed weighted `links = [[u, v, cost], ...]`, or `-1` if any node is unreachable."
        ),
        "p_prime_starter_code": (
            "import heapq\n"
            "def relay_propagation_time(links: list[list[int]], num_nodes: int, source: int) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "import heapq\n"
            "def relay_propagation_time(links: list[list[int]], num_nodes: int, source: int) -> int:\n"
            "    adj = {i: [] for i in range(1, num_nodes + 1)}\n"
            "    for u, v, w in links:\n"
            "        adj[u].append((v, w))\n"
            "    dist = {source: 0}\n"
            "    pq = [(0, source)]\n"
            "    while pq:\n"
            "        d, u = heapq.heappop(pq)\n"
            "        if d > dist.get(u, float('inf')):\n"
            "            continue\n"
            "        for v, w in adj[u]:\n"
            "            if d + w < dist.get(v, float('inf')):\n"
            "                dist[v] = d + w\n"
            "                heapq.heappush(pq, (dist[v], v))\n"
            "    return max(dist.values()) if len(dist) == num_nodes else -1\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([[1, 2, 8], [1, 3, 1], [3, 2, 2]], 3, 1) == 3\n"
            "    assert candidate([[1, 2, 4]], 3, 1) == -1\n"
            "    assert candidate([[1, 2, 2], [2, 3, 3]], 3, 1) == 5\n"
        ),
        "gold_socratic_hint": (
            "Trace your priority queue on the graph `1 -> 2 (cost 10), 1 -> 3 (cost 2), 3 -> 2 (cost 3)`: "
            "when neighbor `2` is first discovered from `1`, `dist[2]` is set to `10`. Later, when `3` is popped "
            "with distance `2`, what prevents your condition `if v not in dist` from relaxing `dist[2]` to the shorter path `2 + 3 = 5`?"
        ),
    },

    # 13. Disjoint Set Union-Find (Connected Components -> Redundant Cluster Count)
    {
        "family_id": "union_find_components",
        "algorithmic_concept": "Disjoint Set Union (Union-Find) linking root representatives root_u != root_v rather than raw nodes",
        "misconception_type": "assigning_parent_of_node_instead_of_root_representative",
        "p_title": "Count Connected Server Clusters",
        "p_entry_point": "count_clusters",
        "p_prompt": (
            "Write a function `count_clusters(n: int, edges: list[list[int]]) -> int` that returns "
            "the number of connected components in an undirected graph with `n` nodes (`0` to `n-1`)."
        ),
        "p_buggy_code": (
            "def count_clusters(n: int, edges: list[list[int]]) -> int:\n"
            "    parent = list(range(n))\n"
            "    def find(x):\n"
            "        while parent[x] != x:\n"
            "            x = parent[x]\n"
            "        return x\n"
            "    components = n\n"
            "    for u, v in edges:\n"
            "        if find(u) != find(v):\n"
            "            parent[u] = v\n"
            "            components -= 1\n"
            "    return components\n"
        ),
        "p_canonical_solution": (
            "def count_clusters(n: int, edges: list[list[int]]) -> int:\n"
            "    parent = list(range(n))\n"
            "    def find(x):\n"
            "        while parent[x] != x:\n"
            "            parent[x] = parent[parent[x]]\n"
            "            x = parent[x]\n"
            "        return x\n"
            "    components = n\n"
            "    for u, v in edges:\n"
            "        ru, rv = find(u), find(v)\n"
            "        if ru != rv:\n"
            "            parent[ru] = rv\n"
            "            components -= 1\n"
            "    return components\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate(5, [[0, 1], [1, 2], [3, 4]]) == 2\n"
            "    assert candidate(3, [[0, 1], [0, 2], [1, 2]]) == 1\n"
            "    assert candidate(4, [[0, 1], [0, 2], [1, 3]]) == 1\n"
            "    assert candidate(3, []) == 3\n"
        ),
        "p_prime_title": "Count Isolated Power Subgrids",
        "p_prime_entry_point": "count_subgrids",
        "p_prime_prompt": (
            "Write a function `count_subgrids(num_stations: int, cables: list[list[int]]) -> int` "
            "that returns the number of connected power subgrids among `0 .. num_stations - 1`."
        ),
        "p_prime_starter_code": (
            "def count_subgrids(num_stations: int, cables: list[list[int]]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def count_subgrids(num_stations: int, cables: list[list[int]]) -> int:\n"
            "    parent = list(range(num_stations))\n"
            "    def find(x):\n"
            "        while parent[x] != x:\n"
            "            parent[x] = parent[parent[x]]\n"
            "            x = parent[x]\n"
            "        return x\n"
            "    comp = num_stations\n"
            "    for a, b in cables:\n"
            "        ra, rb = find(a), find(b)\n"
            "        if ra != rb:\n"
            "            parent[ra] = rb\n"
            "            comp -= 1\n"
            "    return comp\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate(4, [[0, 1], [0, 2], [0, 3]]) == 1\n"
            "    assert candidate(5, [[0, 1], [2, 3]]) == 3\n"
            "    assert candidate(2, [[0, 1], [1, 0]]) == 1\n"
        ),
        "gold_socratic_hint": (
            "Trace your Union-Find on edges `[0, 1]` followed by `[0, 2]`: after `[0, 1]`, `parent[0]` is `1`. "
            "When you process `[0, 2]` and overwrite `parent[0] = 2` directly instead of linking the root `find(0)` "
            "to `find(2)`, what happens to the connection between node `1` and node `0`?"
        ),
    },

    # 14. Backtracking / Combinatorial Search (Combination Sum -> Target Resource Allocation)
    {
        "family_id": "backtracking_combination_sum",
        "algorithmic_concept": "Backtracking Search avoiding duplicate permutations by passing the current start index `i` into recursion",
        "misconception_type": "restarting_loop_from_zero_creates_duplicate_permutations",
        "p_title": "Unique Combinations Summing to Target",
        "p_entry_point": "combination_sum",
        "p_prompt": (
            "Write a function `combination_sum(candidates: list[int], target: int) -> list[list[int]]` "
            "that returns all unique combinations of `candidates` (distinct positive integers, reusable unlimited times) "
            "that sum to `target`, with each combination sorted in non-decreasing order."
        ),
        "p_buggy_code": (
            "def combination_sum(candidates: list[int], target: int) -> list[list[int]]:\n"
            "    candidates.sort()\n"
            "    res = []\n"
            "    def dfs(rem, path):\n"
            "        if rem == 0:\n"
            "            res.append(list(path))\n"
            "            return\n"
            "        for x in candidates:\n"
            "            if x <= rem:\n"
            "                path.append(x)\n"
            "                dfs(rem - x, path)\n"
            "                path.pop()\n"
            "    dfs(target, [])\n"
            "    return res\n"
        ),
        "p_canonical_solution": (
            "def combination_sum(candidates: list[int], target: int) -> list[list[int]]:\n"
            "    candidates.sort()\n"
            "    res = []\n"
            "    def dfs(start_idx, rem, path):\n"
            "        if rem == 0:\n"
            "            res.append(list(path))\n"
            "            return\n"
            "        for i in range(start_idx, len(candidates)):\n"
            "            x = candidates[i]\n"
            "            if x > rem:\n"
            "                break\n"
            "            path.append(x)\n"
            "            dfs(i, rem - x, path)\n"
            "            path.pop()\n"
            "    dfs(0, target, [])\n"
            "    return res\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 3, 6, 7], 7) == [[2, 2, 3], [7]]\n"
            "    assert candidate([2, 3, 5], 8) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]\n"
            "    assert candidate([2], 1) == []\n"
        ),
        "p_prime_title": "Unique Bundle Allocations For Quota",
        "p_prime_entry_point": "bundle_allocations",
        "p_prime_prompt": (
            "Write a function `bundle_allocations(sizes: list[int], quota: int) -> list[list[int]]` "
            "that returns all unique non-decreasing combinations of `sizes` (each usable unlimited times) "
            "whose sum equals `quota`."
        ),
        "p_prime_starter_code": (
            "def bundle_allocations(sizes: list[int], quota: int) -> list[list[int]]:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def bundle_allocations(sizes: list[int], quota: int) -> list[list[int]]:\n"
            "    sizes.sort()\n"
            "    ans = []\n"
            "    def backtrack(idx, rem, curr):\n"
            "        if rem == 0:\n"
            "            ans.append(list(curr))\n"
            "            return\n"
            "        for i in range(idx, len(sizes)):\n"
            "            if sizes[i] > rem:\n"
            "                break\n"
            "            curr.append(sizes[i])\n"
            "            backtrack(i, rem - sizes[i], curr)\n"
            "            curr.pop()\n"
            "    backtrack(0, quota, [])\n"
            "    return ans\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 4, 6], 6) == [[2, 2, 2], [2, 4], [6]]\n"
            "    assert candidate([3, 5], 8) == [[3, 5]]\n"
            "    assert candidate([4], 3) == []\n"
        ),
        "gold_socratic_hint": (
            "Notice what happens when `candidates = [2, 3, 6, 7]` and `target = 7`: because your loop always "
            "restarts over all `x in candidates`, it generates `[2, 2, 3]`, `[2, 3, 2]`, and `[3, 2, 2]` as separate "
            "paths. How can passing a `start_idx` parameter into `dfs` ensure you never pick an element earlier than the current index?"
        ),
    },

    # 15. Bitwise XOR / Parity Invariant (Single Number -> Unpaired Packet ID)
    {
        "family_id": "bitwise_xor_cancellation",
        "algorithmic_concept": "Bitwise XOR Self-Inverse & Commutative Cancellation (a ^ a == 0 and a ^ 0 == a)",
        "misconception_type": "adjacent_pair_comparison_fails_at_last_element",
        "p_title": "Find Unique Unpaired Element",
        "p_entry_point": "single_number",
        "p_prompt": (
            "Write a function `single_number(nums: list[int]) -> int` where every element appears "
            "twice except for one element which appears only once. Return that single element."
        ),
        "p_buggy_code": (
            "def single_number(nums: list[int]) -> int:\n"
            "    nums.sort()\n"
            "    for i in range(0, len(nums) - 1, 2):\n"
            "        if nums[i] != nums[i + 1]:\n"
            "            return nums[i]\n"
            "    return 0\n"
        ),
        "p_canonical_solution": (
            "def single_number(nums: list[int]) -> int:\n"
            "    ans = 0\n"
            "    for x in nums:\n"
            "        ans ^= x\n"
            "    return ans\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 2, 1]) == 1\n"
            "    assert candidate([4, 1, 2, 1, 2]) == 4\n"
            "    assert candidate([1]) == 1\n"
            "    assert candidate([1, 1, 9]) == 9\n"
        ),
        "p_prime_title": "Identify Unacknowledged Request ID",
        "p_prime_entry_point": "unacked_request_id",
        "p_prime_prompt": (
            "Write a function `unacked_request_id(packet_ids: list[int]) -> int` where every request ID "
            "appears twice (sent and acknowledged) except one ID that appears once. Return that ID."
        ),
        "p_prime_starter_code": (
            "def unacked_request_id(packet_ids: list[int]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def unacked_request_id(packet_ids: list[int]) -> int:\n"
            "    res = 0\n"
            "    for pid in packet_ids:\n"
            "        res ^= pid\n"
            "    return res\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([10, 20, 10]) == 20\n"
            "    assert candidate([7]) == 7\n"
            "    assert candidate([3, 8, 3, 9, 8]) == 9\n"
        ),
        "gold_socratic_hint": (
            "What does your loop return when the unique element is the largest number in the sorted array "
            "(for example `[1, 1, 9]` or `[1]`), sitting at the very last index `nums[-1]`? Even better, "
            "which bitwise operator satisfies `x ^ x == 0` and `x ^ 0 == x` so pairs cancel out in O(n) time?"
        ),
    },

    # 16. HumanEvalFix (BigCode HumanEval/0): has_close_elements -> has_ conflicting_timestamps
    {
        "family_id": "humanevalfix_000_close_elements",
        "algorithmic_concept": "Pairwise Distinct-Index Absolute Distance Check (idx != idx2 and abs(elem - elem2) < threshold)",
        "misconception_type": "signed_difference_instead_of_abs_distance",
        "p_title": "HumanEvalFix #0: Has Close Elements",
        "p_entry_point": "has_close_elements",
        "p_prompt": (
            "Check if in given list of `numbers`, are any two numbers closer to each other than "
            "given `threshold`:\n"
            "`def has_close_elements(numbers: list[float], threshold: float) -> bool:`"
        ),
        "p_buggy_code": (
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            "    for idx, elem in enumerate(numbers):\n"
            "        for idx2, elem2 in enumerate(numbers):\n"
            "            if idx != idx2:\n"
            "                distance = elem - elem2\n"
            "                if distance < threshold:\n"
            "                    return True\n"
            "    return False\n"
        ),
        "p_canonical_solution": (
            "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
            "    for idx, elem in enumerate(numbers):\n"
            "        for idx2, elem2 in enumerate(numbers):\n"
            "            if idx != idx2:\n"
            "                distance = abs(elem - elem2)\n"
            "                if distance < threshold:\n"
            "                    return True\n"
            "    return False\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
        ),
        "p_prime_title": "HumanEvalFix Transfer: Has Conflicting Sensor Readings",
        "p_prime_entry_point": "has_close_readings",
        "p_prime_prompt": (
            "Write a function `has_close_readings(readings: list[float], min_gap: float) -> bool` "
            "that returns `True` if any two distinct readings in `readings` differ in magnitude by "
            "strictly less than `min_gap`, or `False` otherwise."
        ),
        "p_prime_starter_code": (
            "def has_close_readings(readings: list[float], min_gap: float) -> bool:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def has_close_readings(readings: list[float], min_gap: float) -> bool:\n"
            "    for i, r1 in enumerate(readings):\n"
            "        for j, r2 in enumerate(readings):\n"
            "            if i != j and abs(r1 - r2) < min_gap:\n"
            "                return True\n"
            "    return False\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([10.0, 20.0, 30.0], 5.0) == False\n"
            "    assert candidate([10.0, 14.2, 30.0], 5.0) == True\n"
            "    assert candidate([5.0, 5.0], 0.01) == True\n"
            "    assert candidate([1.0, 9.0], 2.0) == False\n"
        ),
        "gold_socratic_hint": (
            "Trace what `distance = elem - elem2` produces when `elem = 1.0` and `elem2 = 5.0`: "
            "is a negative number like `-4.0` always `< threshold` even when the two numbers are far apart? "
            "How should you measure unsigned magnitude distance between two numbers?"
        ),
    },

    # 17. HumanEvalFix (BigCode HumanEval/3): below_zero -> account_overdraft_detected
    {
        "family_id": "humanevalfix_003_below_zero",
        "algorithmic_concept": "Running Prefix Balance Invariant (strictly negative `< 0` vs `<= 0` after each update)",
        "misconception_type": "resetting_balance_or_wrong_zero_comparison",
        "p_title": "HumanEvalFix #3: Below Zero Balance",
        "p_entry_point": "below_zero",
        "p_prompt": (
            "You're given a list of deposit and withdrawal `operations` on a bank account that starts "
            "with zero balance. Detect if at any point the balance falls below zero, and at that point "
            "return `True`; otherwise return `False`:\n"
            "`def below_zero(operations: list[int]) -> bool:`"
        ),
        "p_buggy_code": (
            "def below_zero(operations: list[int]) -> bool:\n"
            "    balance = 0\n"
            "    for op in operations:\n"
            "        balance += op\n"
            "        if balance == 0:\n"
            "            return True\n"
            "    return False\n"
        ),
        "p_canonical_solution": (
            "def below_zero(operations: list[int]) -> bool:\n"
            "    balance = 0\n"
            "    for op in operations:\n"
            "        balance += op\n"
            "        if balance < 0:\n"
            "            return True\n"
            "    return False\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([]) == False\n"
            "    assert candidate([1, 2, -3, 1, 2, -3]) == False\n"
            "    assert candidate([1, 2, -4, 5, 6]) == True\n"
            "    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n"
            "    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True\n"
        ),
        "p_prime_title": "HumanEvalFix Transfer: Battery Depletion Detector",
        "p_prime_entry_point": "battery_below_zero",
        "p_prime_prompt": (
            "Write a function `battery_below_zero(deltas: list[int]) -> bool` starting from a charge of `0` "
            "that applies each energy change in `deltas` sequentially and returns `True` as soon as the "
            "running charge drops strictly below `0`, or `False` if it stays non-negative throughout."
        ),
        "p_prime_starter_code": (
            "def battery_below_zero(deltas: list[int]) -> bool:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def battery_below_zero(deltas: list[int]) -> bool:\n"
            "    charge = 0\n"
            "    for d in deltas:\n"
            "        charge += d\n"
            "        if charge < 0:\n"
            "            return True\n"
            "    return False\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([5, -5, 3, -3]) == False\n"
            "    assert candidate([4, -5, 10]) == True\n"
            "    assert candidate([]) == False\n"
            "    assert candidate([-1]) == True\n"
        ),
        "gold_socratic_hint": (
            "Trace `operations = [1, 2, -3]`: when the running `balance` returns to `0`, has the account "
            "actually fallen strictly *below* zero, or does falling below zero require a strict `< 0` inequality?"
        ),
    },

    # 18. HumanEvalFix (BigCode HumanEval/5): intersperse -> interleave_delimiter
    {
        "family_id": "humanevalfix_005_intersperse",
        "algorithmic_concept": "Fencepost / Interleaving Invariant (N-1 delimiters between N elements, trailing element without delimiter)",
        "misconception_type": "missing_final_element_or_trailing_delimiter",
        "p_title": "HumanEvalFix #5: Intersperse Delimiter",
        "p_entry_point": "intersperse",
        "p_prompt": (
            "Insert a number `delimeter` between every two consecutive elements of input list `numbers`:\n"
            "`def intersperse(numbers: list[int], delimeter: int) -> list[int]:`"
        ),
        "p_buggy_code": (
            "def intersperse(numbers: list[int], delimeter: int) -> list[int]:\n"
            "    if not numbers:\n"
            "        return []\n"
            "    result = []\n"
            "    for n in numbers[:-1]:\n"
            "        result.append(n)\n"
            "        result.append(delimeter)\n"
            "    return result\n"
        ),
        "p_canonical_solution": (
            "def intersperse(numbers: list[int], delimeter: int) -> list[int]:\n"
            "    if not numbers:\n"
            "        return []\n"
            "    result = []\n"
            "    for n in numbers[:-1]:\n"
            "        result.append(n)\n"
            "        result.append(delimeter)\n"
            "    result.append(numbers[-1])\n"
            "    return result\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([], 7) == []\n"
            "    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]\n"
            "    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]\n"
            "    assert candidate([9], 4) == [9]\n"
        ),
        "p_prime_title": "HumanEvalFix Transfer: Insert Packet Sync Markers",
        "p_prime_entry_point": "insert_sync_markers",
        "p_prime_prompt": (
            "Write a function `insert_sync_markers(frames: list[int], marker: int) -> list[int]` "
            "that places `marker` between every pair of adjacent elements in `frames` (returning `[]` if `frames` is empty)."
        ),
        "p_prime_starter_code": (
            "def insert_sync_markers(frames: list[int], marker: int) -> list[int]:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def insert_sync_markers(frames: list[int], marker: int) -> list[int]:\n"
            "    if not frames:\n"
            "        return []\n"
            "    out = []\n"
            "    for x in frames[:-1]:\n"
            "        out.append(x)\n"
            "        out.append(marker)\n"
            "    out.append(frames[-1])\n"
            "    return out\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([], 0) == []\n"
            "    assert candidate([10], 99) == [10]\n"
            "    assert candidate([1, 2, 3], 0) == [1, 0, 2, 0, 3]\n"
        ),
        "gold_socratic_hint": (
            "In the fencepost pattern `for n in numbers[:-1]:`, you iterate over the first `N - 1` elements "
            "and append `(n, delimeter)` for each. What happens to the very last fencepost `numbers[-1]` "
            "after the loop finishes, especially when `numbers` has only 1 element?"
        ),
    },

    # 19. MBPP+ (EvalPlus MBPP #56 / Kadane's Algorithm): max_subarray_sum -> max_contiguous_gain
    {
        "family_id": "mbppplus_kadane_max_subarray",
        "algorithmic_concept": "Kadane's 1D Contiguous Subarray DP (initializing cur_sum = max_sum = nums[0] for all-negative support)",
        "misconception_type": "initializing_max_sum_to_zero_fails_on_all_negative_inputs",
        "p_title": "MBPP+ #56: Maximum Contiguous Subarray Sum",
        "p_entry_point": "max_subarray_sum",
        "p_prompt": (
            "Write a function `max_subarray_sum(nums: list[int]) -> int` that finds the maximum "
            "sum of any non-empty contiguous subarray within `nums` (which may contain all negative integers)."
        ),
        "p_buggy_code": (
            "def max_subarray_sum(nums: list[int]) -> int:\n"
            "    best = 0\n"
            "    curr = 0\n"
            "    for x in nums:\n"
            "        curr = max(0, curr + x)\n"
            "        best = max(best, curr)\n"
            "    return best\n"
        ),
        "p_canonical_solution": (
            "def max_subarray_sum(nums: list[int]) -> int:\n"
            "    best = nums[0]\n"
            "    curr = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        curr = max(x, curr + x)\n"
            "        best = max(best, curr)\n"
            "    return best\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n"
            "    assert candidate([-5, -2, -9]) == -2\n"
            "    assert candidate([-7]) == -7\n"
            "    assert candidate([5, 4, -1, 7, 8]) == 23\n"
        ),
        "p_prime_title": "MBPP+ Transfer: Peak Contiguous Thermal Drift",
        "p_prime_entry_point": "max_thermal_drift",
        "p_prime_prompt": (
            "Write a function `max_thermal_drift(drifts: list[int]) -> int` that returns the largest "
            "sum of any non-empty contiguous slice of `drifts` (handling negative-only lists properly)."
        ),
        "p_prime_starter_code": (
            "def max_thermal_drift(drifts: list[int]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def max_thermal_drift(drifts: list[int]) -> int:\n"
            "    best = drifts[0]\n"
            "    curr = drifts[0]\n"
            "    for d in drifts[1:]:\n"
            "        curr = max(d, curr + d)\n"
            "        best = max(best, curr)\n"
            "    return best\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([-8, -3, -6]) == -3\n"
            "    assert candidate([3, -2, 5, -1]) == 6\n"
            "    assert candidate([-4]) == -4\n"
        ),
        "gold_socratic_hint": (
            "What does your function return when every number in `nums` is negative, such as `[-5, -2, -9]`? "
            "If the subarray must be non-empty, how should `curr` and `best` be initialized from `nums[0]`, "
            "and is starting a fresh subarray at `x` represented by `max(x, curr + x)` rather than `max(0, ...)`?"
        ),
    },

    # 20. MBPP+ (EvalPlus MBPP #215): Majority / Boyer-Moore Voting & Hash Frequency
    {
        "family_id": "mbppplus_first_non_repeating",
        "algorithmic_concept": "Two-Pass Frequency Counting Invariant (complete full frequency map before scanning for count == 1)",
        "misconception_type": "checking_frequency_during_single_pass_before_future_duplicates",
        "p_title": "MBPP+ #215: First Non-Repeating Character Index",
        "p_entry_point": "first_uniq_char",
        "p_prompt": (
            "Write a function `first_uniq_char(s: str) -> int` that returns the 0-based index of "
            "the first non-repeating character in string `s`, or `-1` if every character repeats."
        ),
        "p_buggy_code": (
            "def first_uniq_char(s: str) -> int:\n"
            "    counts = {}\n"
            "    for c in s:\n"
            "        counts[c] = counts.get(c, 0) + 1\n"
            "    for i, c in enumerate(s):\n"
            "        if counts[c] > 1:\n"
            "            return i\n"
            "    return -1\n"
        ),
        "p_canonical_solution": (
            "def first_uniq_char(s: str) -> int:\n"
            "    counts = {}\n"
            "    for c in s:\n"
            "        counts[c] = counts.get(c, 0) + 1\n"
            "    for i, c in enumerate(s):\n"
            "        if counts[c] == 1:\n"
            "            return i\n"
            "    return -1\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate('leetcode') == 0\n"
            "    assert candidate('loveleetcode') == 2\n"
            "    assert candidate('aabb') == -1\n"
            "    assert candidate('z') == 0\n"
        ),
        "p_prime_title": "MBPP+ Transfer: First Unique Transaction Code Index",
        "p_prime_entry_point": "first_unique_code",
        "p_prime_prompt": (
            "Write a function `first_unique_code(codes: list[int]) -> int` that returns the 0-based "
            "index of the first element in `codes` that appears exactly once in the entire list, or `-1` if none exists."
        ),
        "p_prime_starter_code": (
            "def first_unique_code(codes: list[int]) -> int:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def first_unique_code(codes: list[int]) -> int:\n"
            "    freq = {}\n"
            "    for x in codes:\n"
            "        freq[x] = freq.get(x, 0) + 1\n"
            "    for idx, x in enumerate(codes):\n"
            "        if freq[x] == 1:\n"
            "            return idx\n"
            "    return -1\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([4, 5, 1, 2, 0, 4]) == 1\n"
            "    assert candidate([7, 7, 8, 8]) == -1\n"
            "    assert candidate([99]) == 0\n"
        ),
        "gold_socratic_hint": (
            "In your second pass over the string, you check whether the character's total frequency "
            "is strictly greater than one: does that condition match a repeating character or a unique "
            "character that occurred only a single time across the string?"
        ),
    },

    # 21. MBPP+ (EvalPlus MBPP #412): Greedy Jump Game Reachability Invariant
    {
        "family_id": "mbppplus_jump_game_reachability",
        "algorithmic_concept": "Greedy Farthest-Reachable Index Invariant (checking i > max_reach before extending max_reach = max(max_reach, i + jump))",
        "misconception_type": "ignoring_unreachable_gap_index_i_greater_than_max_reach",
        "p_title": "MBPP+ #412: Can Reach Last Index",
        "p_entry_point": "can_jump",
        "p_prompt": (
            "Given an integer array `nums` where `nums[i]` represents the maximum jump length from index `i`, "
            "write `can_jump(nums: list[int]) -> bool` returning `True` if you can reach the last index "
            "starting from index `0`, or `False` otherwise."
        ),
        "p_buggy_code": (
            "def can_jump(nums: list[int]) -> bool:\n"
            "    max_reach = 0\n"
            "    for i, step in enumerate(nums):\n"
            "        max_reach = max(max_reach, i + step)\n"
            "    return max_reach >= len(nums) - 1\n"
        ),
        "p_canonical_solution": (
            "def can_jump(nums: list[int]) -> bool:\n"
            "    max_reach = 0\n"
            "    for i, step in enumerate(nums):\n"
            "        if i > max_reach:\n"
            "            return False\n"
            "        max_reach = max(max_reach, i + step)\n"
            "    return True\n"
        ),
        "p_test_code": (
            "def check(candidate):\n"
            "    assert candidate([2, 3, 1, 1, 4]) == True\n"
            "    assert candidate([3, 2, 1, 0, 4]) == False\n"
            "    assert candidate([0]) == True\n"
            "    assert candidate([0, 2, 3]) == False\n"
        ),
        "p_prime_title": "MBPP+ Transfer: Relay Station Signal Reachability",
        "p_prime_entry_point": "can_relay_reach",
        "p_prime_prompt": (
            "Write a function `can_relay_reach(ranges: list[int]) -> bool` where `ranges[i]` is the "
            "maximum number of stations forward station `i` can transmit. Return `True` if station `0` "
            "can reach the final station `len(ranges) - 1`, or `False` otherwise."
        ),
        "p_prime_starter_code": (
            "def can_relay_reach(ranges: list[int]) -> bool:\n"
            "    # Your solution here\n"
            "    pass\n"
        ),
        "p_prime_canonical_solution": (
            "def can_relay_reach(ranges: list[int]) -> bool:\n"
            "    farthest = 0\n"
            "    for i, r in enumerate(ranges):\n"
            "        if i > farthest:\n"
            "            return False\n"
            "        farthest = max(farthest, i + r)\n"
            "    return True\n"
        ),
        "p_prime_test_code": (
            "def check(candidate):\n"
            "    assert candidate([1, 2, 0, 1]) == True\n"
            "    assert candidate([1, 0, 2, 1]) == False\n"
            "    assert candidate([0]) == True\n"
        ),
        "gold_socratic_hint": (
            "Trace your loop on `nums = [0, 2, 3]`: at `i = 0`, `max_reach` is `0`, yet on the next "
            "iteration `i = 1` your loop still executes `i + step` even though index `1` was never reachable! "
            "What condition between `i` and `max_reach` must hold before you are allowed to jump from index `i`?"
        ),
    },
]


def get_seed_paired_problems() -> List[Dict[str, Any]]:
    """Returns a deep copy of the verified seed paired problems."""
    import copy

    return copy.deepcopy(SEED_PAIRED_PROBLEMS)
