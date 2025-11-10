-- Analyze post activity by day of week (0=Sunday, 6=Saturday)
SELECT 
    EXTRACT(DOW FROM "CreationDate") AS day_of_week,
    CASE EXTRACT(DOW FROM "CreationDate")
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_name,
    COUNT(*) AS post_count,
    AVG("Score") AS avg_score
FROM "Posts"
WHERE "CreationDate" IS NOT NULL
GROUP BY day_of_week
ORDER BY day_of_week;
