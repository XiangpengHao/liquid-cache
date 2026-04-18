-- Get questions with titles starting with 'How to' or 'What is'
SELECT "Id", "Title", "Score", "ViewCount", "CreationDate"
FROM "Posts"
WHERE "PostTypeId" = 1
  AND ("Title" LIKE 'How to%' OR "Title" LIKE 'What is%')
ORDER BY "Score" DESC
LIMIT 50;
