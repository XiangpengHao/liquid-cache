-- Cross-year user growth - users active in multiple consecutive years
SELECT 
    EXTRACT(YEAR FROM p."CreationDate") AS "Year",
    COUNT(DISTINCT u."Id") AS "ActiveUsers",
    COUNT(DISTINCT CASE WHEN EXTRACT(YEAR FROM u."CreationDate") = EXTRACT(YEAR FROM p."CreationDate") THEN u."Id" END) AS "NewUsers",
    COUNT(DISTINCT CASE WHEN EXTRACT(YEAR FROM u."CreationDate") < EXTRACT(YEAR FROM p."CreationDate") THEN u."Id" END) AS "ReturningUsers"
FROM "Posts" p
JOIN "Users" u ON p."OwnerUserId" = u."Id"
WHERE p."PostTypeId" IN (1, 2)
  AND p."CreationDate" >= CURRENT_TIMESTAMP - INTERVAL '5 years'
GROUP BY EXTRACT(YEAR FROM p."CreationDate")
ORDER BY "Year" DESC;
