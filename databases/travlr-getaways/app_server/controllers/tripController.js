const Trip = require('../models/Trip');
const { validationResult } = require('express-validator');

// Get all trips with filtering and sorting
const getTrips = async (req, res) => {
  try {
    const { destination, minPrice, maxPrice, sortBy, sortOrder } = req.query;
    
    let query = {};
    
    // Filter by destination (resort)
    if (destination) {
      query.resort = { $regex: destination, $options: 'i' };
    }
    
    // Filter by price range
    if (minPrice || maxPrice) {
      query.perPerson = {};
      if (minPrice) query.perPerson.$gte = Number(minPrice);
      if (maxPrice) query.perPerson.$lte = Number(maxPrice);
    }
    
    // Sort options
    let sort = {};
    if (sortBy) {
      sort[sortBy] = sortOrder === 'desc' ? -1 : 1;
    } else {
      sort.start = 1; // Default sort by start date
    }
    
    const trips = await Trip.find(query).sort(sort);
    res.json(trips);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching trips', error: error.message });
  }
};

// Get single trip by code
const getTrip = async (req, res) => {
  try {
    const trip = await Trip.findOne({ code: req.params.code });
    if (!trip) {
      return res.status(404).json({ message: 'Trip not found' });
    }
    res.json(trip);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching trip', error: error.message });
  }
};

// Create new trip (admin only)
const createTrip = async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const trip = new Trip(req.body);
    await trip.save();
    res.status(201).json(trip);
  } catch (error) {
    if (error.code === 11000) {
      res.status(400).json({ message: 'Trip code already exists' });
    } else {
      res.status(500).json({ message: 'Error creating trip', error: error.message });
    }
  }
};

// Update trip (admin only)
const updateTrip = async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const trip = await Trip.findOneAndUpdate(
      { code: req.params.code },
      req.body,
      { new: true, runValidators: true }
    );
    
    if (!trip) {
      return res.status(404).json({ message: 'Trip not found' });
    }
    
    res.json(trip);
  } catch (error) {
    res.status(500).json({ message: 'Error updating trip', error: error.message });
  }
};

// Delete trip (admin only)
const deleteTrip = async (req, res) => {
  try {
    const trip = await Trip.findOneAndDelete({ code: req.params.code });
    if (!trip) {
      return res.status(404).json({ message: 'Trip not found' });
    }
    res.json({ message: 'Trip deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Error deleting trip', error: error.message });
  }
};

module.exports = {
  getTrips,
  getTrip,
  createTrip,
  updateTrip,
  deleteTrip
};
