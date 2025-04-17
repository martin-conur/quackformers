load 'build/debug/quackformers.duckdb_extension';

CREATE TEMP TABLE EMBEDDINGS AS
SELECT * from embed('How do I register for a new college term?');

SELECT column0::FLOAT[384] FROM EMBEDDINGS;