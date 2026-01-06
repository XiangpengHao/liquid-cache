-- Analyze post activity by day of week (0=Sunday, 6=Saturday)
SELECT
    EXTRACT(DOW FROM "CreationDate") AS day_of_week,
    COUNT(*) AS post_count,
FROM "Posts"
WHERE "CreationDate" IS NOT NULL
GROUP BY day_of_week
ORDER BY day_of_week;
