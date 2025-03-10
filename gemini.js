import { GoogleGenerativeAI } from "@google/generative-ai";
import pg from 'pg';
import dotenv from 'dotenv';
import express from 'express';
import { pipeline } from '@xenova/transformers';
import pgvector from 'pgvector';

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
// LLMs Model (Gemini)
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
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
        `SELECT id, title, genres, 1 - (embedding <=> $1) AS similarity
             FROM documents
             ORDER BY similarity DESC
        `,
        [pgvector.toSql(queryEmbedding)]
    );
    return result.rows;
}

// API: ค้นหาภาพยนตร์ที่คล้ายกัน
app.get('/gemini/search_movies', async (req, res) => {
    try {
        const question = req.query.query || "Summarize, recommend movie titles that are dramas?, it so popular";
        const retrievedDocs = await retrieveDocuments(question);
        // Prepare context for Gemini
        const context = retrievedDocs.map(doc => `${doc.title} - ${doc.genres}`).join("\n");
        // Create the prompt for Gemini
        const prompt = `
            You are an Movie assistant specialist.

            data: ${context}

            question: ${question}
        `;
        
        // Call the Gemini model to generate ax response
        const result = await model.generateContent(prompt);
        res.status(200).send(result.response.text())
    } catch (error) {
        console.error('Search Error:', error);
        res.status(500).send({ error: 'Error searching movies' });
    }
});

// Start Server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));