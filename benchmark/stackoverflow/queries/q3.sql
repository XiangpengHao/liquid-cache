-- Get all posts created in 2020, grouped by month
SELECT 
    EXTRACT(YEAR FROM "CreationDate") AS year,
    EXTRACT(MONTH FROM "CreationDate") AS month,
    COUNT(*) AS post_count,
    AVG("Score") AS avg_score
FROM "Posts"
WHERE EXTRACT(YEAR FROM "CreationDate") = 2020
GROUP BY year, month
ORDER BY year, month;
