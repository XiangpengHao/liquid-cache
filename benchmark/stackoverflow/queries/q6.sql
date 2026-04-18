-- Comment timing patterns - posts with late-night comment activity
SELECT 
    p."Id",
    p."Title",
    p."CreationDate" AS "PostCreated",
    COUNT(c."Id") AS "TotalComments",
    COUNT(CASE WHEN EXTRACT(HOUR FROM c."CreationDate") BETWEEN 22 AND 23 
               OR EXTRACT(HOUR FROM c."CreationDate") BETWEEN 0 AND 5 THEN 1 END) AS "LateNightComments",
    ROUND(100.0 * COUNT(CASE WHEN EXTRACT(HOUR FROM c."CreationDate") BETWEEN 22 AND 23 
                             OR EXTRACT(HOUR FROM c."CreationDate") BETWEEN 0 AND 5 THEN 1 END) / COUNT(c."Id"), 2) AS "LateNightPercentage"
FROM "Posts" p
JOIN "Comments" c ON p."Id" = c."PostId"
WHERE p."PostTypeId" = 1
  AND p."CommentCount" >= 10
  AND c."CreationDate" >= CURRENT_TIMESTAMP - INTERVAL '18 months'::interval
GROUP BY p."Id", p."Title", p."CreationDate"
HAVING COUNT(CASE WHEN EXTRACT(HOUR FROM c."CreationDate") BETWEEN 22 AND 23 
                   OR EXTRACT(HOUR FROM c."CreationDate") BETWEEN 0 AND 5 THEN 1 END) >= 5
ORDER BY "LateNightPercentage" DESC
LIMIT 40;
