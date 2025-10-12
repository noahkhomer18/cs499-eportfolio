const Review = require('../models/Review');
const Trip = require('../models/Trip');
const { validationResult } = require('express-validator');

// Get all reviews for a trip
const getReviewsByTrip = async (req, res) => {
  try {
    const reviews = await Review.find({ tripId: req.params.tripId })
      .populate('userId', 'name')
      .sort({ dateCreated: -1 });
    res.json(reviews);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching reviews', error: error.message });
  }
};

// Get all reviews by a user
const getReviewsByUser = async (req, res) => {
  try {
    const reviews = await Review.find({ userId: req.user._id })
      .populate('tripId', 'name code')
      .sort({ dateCreated: -1 });
    res.json(reviews);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching reviews', error: error.message });
  }
};

// Create new review
const createReview = async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    // Check if trip exists
    const trip = await Trip.findById(req.body.tripId);
    if (!trip) {
      return res.status(404).json({ message: 'Trip not found' });
    }

    // Check if user already reviewed this trip
    const existingReview = await Review.findOne({
      tripId: req.body.tripId,
      userId: req.user._id
    });

    if (existingReview) {
      return res.status(400).json({ message: 'You have already reviewed this trip' });
    }

    const review = new Review({
      ...req.body,
      userId: req.user._id
    });

    await review.save();
    await review.populate('userId', 'name');
    res.status(201).json(review);
  } catch (error) {
    res.status(500).json({ message: 'Error creating review', error: error.message });
  }
};

// Update review (only by the user who created it)
const updateReview = async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const review = await Review.findOneAndUpdate(
      { _id: req.params.id, userId: req.user._id },
      req.body,
      { new: true, runValidators: true }
    ).populate('userId', 'name');

    if (!review) {
      return res.status(404).json({ message: 'Review not found or you are not authorized to update it' });
    }

    res.json(review);
  } catch (error) {
    res.status(500).json({ message: 'Error updating review', error: error.message });
  }
};

// Delete review (only by the user who created it)
const deleteReview = async (req, res) => {
  try {
    const review = await Review.findOneAndDelete({
      _id: req.params.id,
      userId: req.user._id
    });

    if (!review) {
      return res.status(404).json({ message: 'Review not found or you are not authorized to delete it' });
    }

    res.json({ message: 'Review deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Error deleting review', error: error.message });
  }
};

module.exports = {
  getReviewsByTrip,
  getReviewsByUser,
  createReview,
  updateReview,
  deleteReview
};
