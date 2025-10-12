const mongoose = require('mongoose');

const tripSchema = new mongoose.Schema({
  code: {
    type: String,
    required: [true, 'Trip code is required'],
    unique: true,
    uppercase: true,
    trim: true
  },
  name: {
    type: String,
    required: [true, 'Trip name is required'],
    trim: true,
    maxlength: [100, 'Trip name cannot be more than 100 characters']
  },
  length: {
    type: Number,
    required: [true, 'Trip length is required'],
    min: [1, 'Trip length must be at least 1 day']
  },
  start: {
    type: Date,
    required: [true, 'Start date is required']
  },
  resort: {
    type: String,
    required: [true, 'Resort is required'],
    trim: true,
    maxlength: [100, 'Resort name cannot be more than 100 characters']
  },
  perPerson: {
    type: Number,
    required: [true, 'Price per person is required'],
    min: [0, 'Price cannot be negative']
  },
  image: {
    type: String,
    trim: true
  },
  description: {
    type: String,
    trim: true,
    maxlength: [500, 'Description cannot be more than 500 characters']
  }
}, {
  timestamps: true
});

// Index for better query performance
tripSchema.index({ code: 1 });
tripSchema.index({ start: 1 });
tripSchema.index({ perPerson: 1 });

module.exports = mongoose.model('Trip', tripSchema);
