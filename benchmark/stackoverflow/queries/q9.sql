-- "Controversial" posts with a close upvote/downvote ratio
-- These are posts that sharply divide community opinion.
SELECT
    p."Id",
    p."Title",
    SUM(CASE WHEN v."VoteTypeId" = 2 THEN 1 ELSE 0 END) AS UpVotes,
    SUM(CASE WHEN v."VoteTypeId" = 3 THEN 1 ELSE 0 END) AS DownVotes
FROM "Posts" p
JOIN "Votes" v ON p."Id" = v."PostId"
WHERE v."VoteTypeId" IN (2, 3) -- UpMod, DownMod
GROUP BY p."Id", p."Title"
HAVING COUNT(*) > 100
   AND (CAST(SUM(CASE WHEN v."VoteTypeId" = 2 THEN 1 ELSE 0 END) AS DECIMAL) / COUNT(*)) BETWEEN 0.4 AND 0.6
ORDER BY COUNT(*) DESC;
