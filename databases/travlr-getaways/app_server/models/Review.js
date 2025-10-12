const mongoose = require('mongoose');

const reviewSchema = new mongoose.Schema({
  tripId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Trip',
    required: [true, 'Trip ID is required']
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: [true, 'User ID is required']
  },
  rating: {
    type: Number,
    required: [true, 'Rating is required'],
    min: [1, 'Rating must be at least 1'],
    max: [5, 'Rating cannot be more than 5']
  },
  comment: {
    type: String,
    required: [true, 'Comment is required'],
    trim: true,
    maxlength: [500, 'Comment cannot be more than 500 characters']
  },
  dateCreated: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Index for better query performance
reviewSchema.index({ tripId: 1 });
reviewSchema.index({ userId: 1 });
reviewSchema.index({ rating: 1 });

// Ensure one review per user per trip
reviewSchema.index({ tripId: 1, userId: 1 }, { unique: true });

module.exports = mongoose.model('Review', reviewSchema);
