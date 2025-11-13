-- Q3 - When do people use BlueSky
SELECT
    variant_get(data, 'commit.collection', 'string') AS event,
    EXTRACT(HOUR FROM TO_TIMESTAMP_MICROS(variant_get(data, 'time_us', 'bigint'))) AS hour_of_day,
    COUNT(*) AS count
FROM bluesky
WHERE variant_get(data, 'kind', 'string') = 'commit'
  AND variant_get(data, 'commit.operation', 'string') = 'create'
  AND variant_get(data, 'commit.collection', 'string') IN ('app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like')
GROUP BY event, hour_of_day
ORDER BY hour_of_day, event;
