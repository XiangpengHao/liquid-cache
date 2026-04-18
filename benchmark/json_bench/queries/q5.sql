-- Q5 - top 3 users with longest activity
SELECT
    variant_get(data, 'did', 'Utf8') AS user_id,
    CAST(EXTRACT(EPOCH FROM (MAX(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'Int64'))) - MIN(TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'Int64'))))) * 1000 AS BIGINT) AS activity_span
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC
LIMIT 3;
