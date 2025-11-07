-- Users who answered their own questions and accepted their own answer
-- This can indicate self-reliance or potentially "gaming" the reputation system.
SELECT
    p."OwnerUserId",
    u."DisplayName",
    COUNT(*) AS SelfAnsweredCount
FROM "Posts" p
JOIN "Users" u ON p."OwnerUserId" = u."Id"
JOIN "Posts" a ON p."AcceptedAnswerId" = a."Id"
WHERE p."PostTypeId" = 1 -- Question
  AND p."OwnerUserId" = a."OwnerUserId"
GROUP BY p."OwnerUserId", u."DisplayName"
HAVING COUNT(*) > 1
ORDER BY SelfAnsweredCount DESC;