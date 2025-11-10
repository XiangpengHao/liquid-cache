-- Get all posts grouped by month
SELECT 
    EXTRACT(MONTH FROM "CreationDate") AS month,
    COUNT(*) AS post_count,
    AVG("Score") AS avg_score
FROM "Posts"
GROUP BY month
ORDER BY month;
