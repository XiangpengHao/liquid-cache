-- Users with suspicious voting patterns - high downvote ratio on specific days
SELECT 
    u."Id",
    u."DisplayName",
    u."UpVotes",
    u."DownVotes",
    ROUND(u."DownVotes"::numeric / NULLIF(u."UpVotes", 0), 2) AS "DownvoteRatio",
    u."Reputation"
FROM "Users" u
WHERE u."DownVotes" > 100
  AND u."DownVotes"::numeric / NULLIF(u."UpVotes", 0) > 0.5
  AND EXTRACT(DOW FROM u."LastAccessDate") IN (1, 5)  -- Mondays or Fridays
  AND u."Reputation" > 1000
ORDER BY "DownvoteRatio" DESC
LIMIT 30;
