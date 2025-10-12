const express = require('express');
const router = express.Router();
const { body } = require('express-validator');
const { auth, requireAdmin } = require('../middleware/auth');
const tripController = require('../controllers/tripController');
const reviewController = require('../controllers/reviewController');
const authController = require('../controllers/authController');

// Trip validation rules
const tripValidation = [
  body('code').notEmpty().withMessage('Trip code is required'),
  body('name').notEmpty().withMessage('Trip name is required'),
  body('length').isNumeric().withMessage('Trip length must be a number'),
  body('start').isISO8601().withMessage('Start date must be valid'),
  body('resort').notEmpty().withMessage('Resort is required'),
  body('perPerson').isNumeric().withMessage('Price per person must be a number')
];

// Review validation rules
const reviewValidation = [
  body('tripId').isMongoId().withMessage('Valid trip ID is required'),
  body('rating').isInt({ min: 1, max: 5 }).withMessage('Rating must be between 1 and 5'),
  body('comment').notEmpty().withMessage('Comment is required')
];

// Auth validation rules
const authValidation = [
  body('name').notEmpty().withMessage('Name is required'),
  body('email').isEmail().withMessage('Valid email is required'),
  body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters')
];

const loginValidation = [
  body('email').isEmail().withMessage('Valid email is required'),
  body('password').notEmpty().withMessage('Password is required')
];

// Auth routes
router.post('/register', authValidation, authController.register);
router.post('/login', loginValidation, authController.login);
router.get('/me', auth, authController.getMe);

// Trip routes
router.get('/trips', tripController.getTrips);
router.get('/trips/:code', tripController.getTrip);
router.post('/trips', auth, requireAdmin, tripValidation, tripController.createTrip);
router.put('/trips/:code', auth, requireAdmin, tripValidation, tripController.updateTrip);
router.delete('/trips/:code', auth, requireAdmin, tripController.deleteTrip);

// Review routes
router.get('/trips/:tripId/reviews', reviewController.getReviewsByTrip);
router.get('/reviews/my', auth, reviewController.getReviewsByUser);
router.post('/reviews', auth, reviewValidation, reviewController.createReview);
router.put('/reviews/:id', auth, reviewValidation, reviewController.updateReview);
router.delete('/reviews/:id', auth, reviewController.deleteReview);

module.exports = router;
