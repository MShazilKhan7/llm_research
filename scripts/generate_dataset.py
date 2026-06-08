"""
generate_dataset.py
Generates a synthetic ground truth dataset of 200 software issues.

The random seed is configurable and recorded in dataset/dataset_meta.json
so the exact dataset can always be regenerated identically.

Usage
-----
  python scripts/generate_dataset.py                  # seed=42, n=200
  python scripts/generate_dataset.py --seed 7 --n 300
"""

import csv
import json
import random
import os
import argparse
from datetime import datetime, timezone

ISSUE_TEMPLATES = [
    ("Fix login bug", "Users cannot log in sometimes. Please fix it."),
    ("Update UI", "The interface needs updating."),
    ("Add export feature", "We need to export data to CSV format from the reports page."),
    ("Performance issue", "The app is slow."),
    ("Null pointer exception", "NullPointerException occurs in UserService.java at line 142 when userId is null."),
    ("Search not working", "Search returns no results."),
    ("Add dark mode", "Implement dark mode toggle in settings with persistent preference via localStorage."),
    ("Crash on startup", "App crashes when opened."),
    ("Email validation", "Validate email format on registration form and show inline error messages."),
    ("Database timeout", "DB queries timeout after 30 seconds under heavy load with more than 500 concurrent users."),
    ("Missing error handling", "Handle errors properly."),
    ("API rate limiting", "Implement rate limiting of 100 requests per minute per API key with 429 responses."),
    ("Button color", "Change the button color."),
    ("Memory leak", "Memory usage grows unbounded in the background sync service over 24 hours."),
    ("Update docs", "Documentation is outdated."),
    ("File upload fails", "Uploading files larger than 10MB fails with a 413 error on the /upload endpoint."),
    ("Wrong total", "The total is wrong in the cart."),
    ("Session expiry", "Implement automatic session expiry after 30 minutes of inactivity with a warning dialog."),
    ("Fix typo", "There's a typo somewhere."),
    ("Pagination broken", "Pagination does not work when filter is applied on the user management table."),
    ("Add logging", "Add proper logging."),
    ("Two-factor auth", "Add TOTP-based 2FA support for admin accounts with backup codes."),
    ("Improve something", "Something needs to be improved."),
    ("CSV import fails", "Importing CSV files with special characters (commas in fields) fails silently."),
    ("Profile page slow", "The profile page takes 8+ seconds to load due to unoptimized image loading."),
    ("Notifications broken", "Push notifications are not delivered to Android devices running Android 13+."),
    ("Timezone bug", "Timestamps display in UTC instead of user's local timezone in the event log."),
    ("Dropdown empty", "Country dropdown is empty on the registration form for new users."),
    ("Report generation", "The monthly report generation times out for datasets with more than 10,000 rows."),
    ("Missing feature", "Add the thing we discussed in the meeting."),
    ("Concurrent edit conflict", "Two users editing the same record simultaneously causes data loss silently."),
    ("Wrong redirect", "After logout, users are redirected to dashboard instead of the login page."),
    ("Image resize", "Profile images are not resized/compressed on upload, causing slow page loads."),
    ("Filter not saving", "Applied filters are reset when navigating back to the list view."),
    ("Audit log missing", "Admin actions are not being recorded in the audit log table."),
    ("Language switch broken", "Switching language mid-session reverts to English after page refresh."),
    ("Role permission", "Users with 'viewer' role can access admin endpoints via direct URL."),
    ("Broken link", "There is a broken link on the page."),
    ("Chart rendering", "Bar chart renders incorrectly when data contains negative values."),
    ("Delete confirmation", "Add a confirmation dialog before deleting records permanently."),
]


def generate_ground_truth(
    n: int = 200,
    seed: int = 42,
    output_path: str = "dataset/ground_truth.csv",
) -> str:
    """
    Generate synthetic ground truth CSV.
    Labels are assigned with random.seed(seed) so the output is
    100% reproducible given the same seed and n.
    """
    rng = random.Random(seed)   # isolated RNG — does not affect global state
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    for i in range(1, n + 1):
        template = ISSUE_TEMPLATES[(i - 1) % len(ISSUE_TEMPLATES)]
        title, description = template
        title = f"{title} #{i}"
        rows.append({
            "issue_id":    f"ISSUE-{i:04d}",
            "title":       title,
            "description": description,
            "ambiguous":   rng.randint(0, 1),
            "incomplete":  rng.randint(0, 1),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["issue_id", "title", "description", "ambiguous", "incomplete"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ── Save metadata so the dataset can always be exactly regenerated ──
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_issues":     n,
        "seed":         seed,
        "output_path":  output_path,
        "label_counts": {
            "ambiguous_1":    sum(r["ambiguous"]   for r in rows),
            "ambiguous_0":    sum(1 - r["ambiguous"]   for r in rows),
            "incomplete_1":   sum(r["incomplete"]  for r in rows),
            "incomplete_0":   sum(1 - r["incomplete"]  for r in rows),
        },
        "note": (
            "Re-run generate_dataset.py with the same --seed and --n "
            "to reproduce this exact file."
        ),
    }
    meta_path = os.path.join(os.path.dirname(output_path) or ".", "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Ground truth saved to '{output_path}'  (n={n}, seed={seed})")
    print(f"    Ambiguous  : {meta['label_counts']['ambiguous_1']} positive / "
          f"{meta['label_counts']['ambiguous_0']} negative")
    print(f"    Incomplete : {meta['label_counts']['incomplete_1']} positive / "
          f"{meta['label_counts']['incomplete_0']} negative")
    print(f"[✓] Metadata   saved to '{meta_path}'")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic ground truth dataset")
    p.add_argument("--n",    type=int, default=200,  help="Number of issues (default: 200)")
    p.add_argument("--seed", type=int, default=42,   help="Random seed (default: 42)")
    p.add_argument("--output", default="dataset/ground_truth.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_ground_truth(n=args.n, seed=args.seed, output_path=args.output)
