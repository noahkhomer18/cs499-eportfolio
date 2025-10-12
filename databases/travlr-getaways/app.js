const express = require('express');
const path = require('path');
const cors = require('cors');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Database connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/travlr', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// View engine setup
app.set('view engine', 'hbs');
app.set('views', path.join(__dirname, 'app_server/views'));
app.use(express.static(path.join(__dirname, 'public')));

// Routes
const travellerRouter = require('./app_server/routes/traveller');
const apiRouter = require('./app_server/routes/api');
const { auth, requireAdmin } = require('./app_server/middleware/auth');
const adminController = require('./app_server/controllers/adminController');

app.use('/traveller', travellerRouter);
app.use('/api', apiRouter);

// Admin stats route
app.get('/api/admin/stats', auth, requireAdmin, adminController.getStats);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log("Server running on http://localhost:" + PORT);
});
