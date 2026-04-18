-- Q2 - Top event types together with unique users per event type
SELECT
    variant_get(data, 'commit.collection', 'Utf8') AS event,
    COUNT(*) AS count,
    COUNT(DISTINCT variant_get(data, 'did', 'Utf8')) AS users
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8') = 'create'
GROUP BY event
ORDER BY count DESC;

