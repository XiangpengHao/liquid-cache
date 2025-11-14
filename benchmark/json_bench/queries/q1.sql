-- Q1 - Top event types
SELECT
    variant_get(data, 'commit.collection', 'string') AS event,
    COUNT(*) AS count
FROM bluesky
GROUP BY event
ORDER BY count DESC;

