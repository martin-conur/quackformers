INSTALL quackformers FROM  community;
LOAD quackformers;

INSTALL vss;
LOAD vss;

-- embed function uses vanilla BERT, embed_jina uses Jina implementation (768 output dim)
-- at this moment, only embed function is available in community. To use embed_jina, 
-- you'll need to build the extension locally (see README.md)
CREATE TABLE vector_table AS
SELECT *, embed(text)::FLOAT[384] as embedded_text FROM read_csv_auto('chunks.csv');
CREATE INDEX hnsw_index on vector_table USING HNSW (embedded_text);


-- GETTING MOST IMPORTANT CHUNKS BASED ON QUESTION
SELECT text FROM vector_table
ORDER BY array_distance(embedded_text, embed('In which year was the first book published?')::FLOAT[384])
LIMIT 5;


SELECT text FROM vector_table
ORDER BY array_distance(embedded_text, embed('What is the plot of the Lord of the Rings book?')::FLOAT[384])
LIMIT 5;


SELECT text FROM vector_table
ORDER BY array_distance(embedded_text, embed('What are the critics of the book?')::FLOAT[384])
LIMIT 5;