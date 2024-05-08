import express from "express";
import { question } from "./quest.mjs";
const app = express().get("/:id", (req, res) => {
  const params = req.params;
  res.json(question()[params.id]);
});

app.listen(3000, () => {
  console.log("Server is Running....!");
});
