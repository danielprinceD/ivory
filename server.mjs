import express from "express";
import { question } from "./quest.mjs";
import { body, validationResult } from 'express-validator'

const app = express();

app.use(express.json())

app.post("/:id", (req, res) => {
  const { body: { secret }, params: { id } } = req;
  if (!secret || secret != "codeword") return res.sendStatus(400);
  else {
    res.json(question()[id]);
  }
});

app.listen(3000, () => {
  console.log("Server is Running....!");
});
