-- Q2 - Top event types together with unique users per event type
SELECT
    variant_get(data, 'commit.collection', 'string') AS event,
    COUNT(*) AS count,
    COUNT(DISTINCT variant_get(data, 'did', 'string')) AS users
FROM bluesky
WHERE variant_get(data, 'kind', 'string') = 'commit'
  AND variant_get(data, 'commit.operation', 'string') = 'create'
GROUP BY event
ORDER BY count DESC;

