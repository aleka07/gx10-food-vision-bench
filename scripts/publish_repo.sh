#!/bin/bash
# Create the public GitHub repository and push the current tree.
# The token is read from the managed secret file and never printed.
# Note: filename kept as a technical helper; it is safe to publish (no secrets).
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="gx10-food-vision-bench"
OWNER="aleka07"
TOKEN_FILE="$HOME/.hermes/secrets/github/oauth-token"
TOK="$(tr -d '\n\r' < "$TOKEN_FILE")"
export TOK   # the credential helper runs in a child shell and needs it in the environment

git add -A
git -c user.email="alikhan.amirkhan@gmail.com" -c user.name="Alikhan Amirkhanov" \
    commit -q -m "${1:-Update measurements and article assets}" || echo "nothing to commit"
git branch -M main

if curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOK" \
        "https://api.github.com/repos/$OWNER/$REPO" | grep -q '^200$'; then
    echo "repo already exists: $OWNER/$REPO"
else
    curl -sS -X POST -H "Authorization: Bearer $TOK" -H "Accept: application/vnd.github+json" \
        https://api.github.com/user/repos \
        -d "{\"name\":\"$REPO\",\"description\":\"Food-101 inference benchmark on ARM64 + NVIDIA GB10 (ASUS Ascent GX10): latency, throughput and energy measurements, code and raw data.\",\"private\":false,\"has_issues\":true,\"has_wiki\":false}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('repo:', d.get('full_name'), d.get('html_url'), 'private=', d.get('private'), d.get('message',''))"
fi

git remote remove publish 2>/dev/null || true
git remote add publish "https://github.com/$OWNER/$REPO.git"
git -c credential.helper='!f() { echo username=x-access-token; echo "password=$TOK"; }; f' push -q publish main
git remote remove publish
echo "pushed"
