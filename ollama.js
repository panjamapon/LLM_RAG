import ollama from 'ollama'
import pg from 'pg';
import dotenv from 'dotenv';
import express from 'express';
import { pipeline } from '@xenova/transformers';
import pgvector from 'pgvector';
import z from 'zod';
import zodToJsonSchema from 'zod-to-json-schema'
dotenv.config();
const { Pool } = pg;
const app = express();
const PORT = 8888;

// ใช้ Express Middleware เพื่อให้รับ JSON ได้
app.use(express.json());

// เชื่อมต่อ PostgreSQL
const pool = new Pool({
    user: process.env.USERNAME,
    host: process.env.HOST,
    database: process.env.DATABASE,
    password: process.env.PASSWORD,
    port: process.env.PORT,
});

// Embedding Model
const generateEmbedding = await pipeline('feature-extraction', 'nomic-ai/nomic-embed-text-v1.5', {
    dtype: 'fp16'
});
// ฟังก์ชันดึงข้อมูลจาก Database
const fetchData = async () => {
    const result = await pool.query(
        'SELECT show_id, title, listed_in as genres FROM movies'
    );
    return result.rows;
};

// API: Insert Embedding ลง Database
app.post('/ingest_movies', async (req, res) => {
    try {
        const dataset = await fetchData();

        for (const movie of dataset) {
            const { title, genres } = movie;
            const content = `${title} - ${genres}`;

            // สร้าง Embedding Vector
            const output = await generateEmbedding(content, { pooling: 'mean', normalize: true });
            const embedding = Array.from(output.data);

            // Insert ลง PostgreSQL
            await pool.query(
                `INSERT INTO documents (id, title, genres, embedding) 
                 VALUES ($1, $2, $3, $4)`,
                [movie.show_id, title, genres, pgvector.toSql(embedding)]
            );
        }

        res.status(200).send({ message: 'Data Ingested Successfully' });
    } catch (error) {
        console.error('Error:', error);
        res.status(500).send({ error: 'Error inserting data' });
    }
});

async function retrieveDocuments(query) {
    const output = await generateEmbedding(query, { pooling: 'mean', normalize: true });
    const queryEmbedding = Array.from(output.data);

    // ค้นหาใน PostgreSQL โดยใช้ Cosine Similarity
    const result = await pool.query(
        `SELECT id, title, genres, 10 - (embedding <=> $1) AS similarity
             FROM documents
             ORDER BY similarity DESC
        `,
        [pgvector.toSql(queryEmbedding)]
    );
    return result.rows;
}

const schema = z.object({
    movieName: z.string().describe('Name of the movie cannot be empty'),
    imageUrl: z.string().url().describe('image URL of Movies from imdb'),
    genres: z.array(z.string()).min(1, "At least one genre is required").describe('Genres of the movie'),
});

// API: ค้นหาภาพยนตร์ที่คล้ายกัน
app.get('/ollama/search_movies', async (req, res) => {
    try {
        let question = req.query.query || "can you recommend the greatest movies of all time?";
        const retrievedDocs = await retrieveDocuments(question);
        // Prepare context for Gemini
        const context = retrievedDocs.map(doc => `${doc.title} - ${doc.genres}`).join("\n");
        // LLMs Model (Ollama)
        const response = await ollama.chat({
            model: 'llama3.2',
            messages: [
                {
                    role: 'system', content: context
                },
                {
                    role: 'user', content: question
                }
            ],
            format: zodToJsonSchema(schema)
        })
        // ส่งค่ากลับไปยัง client
        res.status(200).send(response?.message.content);
    } catch (error) {
        console.error('Search Error:', error);
        res.status(500).send({ error: 'Error searching movies' });
    }
});

// Start Server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
