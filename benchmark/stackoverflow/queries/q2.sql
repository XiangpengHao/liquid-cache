-- Get posts with 'CC BY-SA 3.0' license that were created after 2015
SELECT "Id", "Title", "PostTypeId", "CreationDate", "Score"
FROM "Posts"
WHERE "ContentLicense" = 'CC BY-SA 3.0'
  AND "CreationDate" > '2015-01-01'
ORDER BY "CreationDate" DESC
LIMIT 100;
