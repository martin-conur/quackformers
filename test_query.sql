LOAD 'build/debug/quackformers.duckdb_extension';
CREATE TEMP TABLE QUESTIONS(random_questions) AS
VALUES
    ('What is the capital of France?'),
    ('How does a car engine work?'),
    ('What is the tallest mountain in the world?'),
    ('How do airplanes stay in the air?'),
    ('What is the speed of light?'),
    ('Who wrote "To Kill a Mockingbird"?'),
    ('What is the chemical formula for water?'),
    ('How do I bake a chocolate cake?'),
    ('What is the population of Japan?'),
    ('How does photosynthesis work?'),
    ('What is the currency of Brazil?'),
    ('Who painted the Mona Lisa?'),
    ('What is the boiling point of water?'),
    ('How do I play the guitar?'),
    ('What is the largest ocean on Earth?'),
    ('Who discovered gravity?'),
    ('What is the process of making glass?'),
    ('How do I learn a new language?'),
    ('What is the history of the internet?'),
    ('How does a computer processor work?')
;

SELECT embed(RANDOM_QUESTIONS)::FLOAT[384] embedded_questions FROM QUESTIONS;