-- Comments containing only a link
-- Similar to link-only answers, these comments lack context and are often not useful.
SELECT
    "Id",
    "PostId",
    "UserId",
    "Text"
FROM "Comments"
WHERE "Text" LIKE 'http%' AND "Text" NOT LIKE '% %'
ORDER BY "Id" DESC
LIMIT 10;
