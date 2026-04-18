-- High-score questions from 2023 with many answers but no accepted answer
-- These might be interesting, unresolved problems that have attracted a lot of attention.
SELECT
    "Id",
    "Title",
    "Score",
    "AnswerCount",
    "CreationDate"
FROM "Posts"
WHERE "PostTypeId" = 1 -- Question
  AND "Score" > 100
  AND "AnswerCount" > 10
  AND "AcceptedAnswerId" IS NULL
  AND EXTRACT(YEAR FROM "CreationDate") = 2020
ORDER BY "Score" DESC
LIMIT 10;
