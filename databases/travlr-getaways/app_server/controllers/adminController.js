const Review = require('../models/Review');
const Trip = require('../models/Trip');
const User = require('../models/User');

// Get admin statistics using aggregation
const getStats = async (req, res) => {
  try {
    // Get total trips, users, and reviews
    const [totalTrips, totalUsers, totalReviews] = await Promise.all([
      Trip.countDocuments(),
      User.countDocuments(),
      Review.countDocuments()
    ]);

    // Get average rating using aggregation
    const avgRatingResult = await Review.aggregate([
      {
        $group: {
          _id: null,
          averageRating: { $avg: '$rating' },
          totalReviews: { $sum: 1 }
        }
      }
    ]);

    // Get rating distribution
    const ratingDistribution = await Review.aggregate([
      {
        $group: {
          _id: '$rating',
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Get most reviewed trips
    const mostReviewedTrips = await Review.aggregate([
      {
        $group: {
          _id: '$tripId',
          reviewCount: { $sum: 1 },
          averageRating: { $avg: '$rating' }
        }
      },
      { $sort: { reviewCount: -1 } },
      { $limit: 5 },
      {
        $lookup: {
          from: 'trips',
          localField: '_id',
          foreignField: '_id',
          as: 'trip'
        }
      },
      { $unwind: '$trip' },
      {
        $project: {
          tripName: '$trip.name',
          tripCode: '$trip.code',
          reviewCount: 1,
          averageRating: { $round: ['$averageRating', 2] }
        }
      }
    ]);

    // Get recent reviews
    const recentReviews = await Review.find()
      .populate('tripId', 'name code')
      .populate('userId', 'name')
      .sort({ dateCreated: -1 })
      .limit(10);

    const stats = {
      totals: {
        trips: totalTrips,
        users: totalUsers,
        reviews: totalReviews
      },
      averageRating: avgRatingResult.length > 0 ? avgRatingResult[0].averageRating : 0,
      ratingDistribution,
      mostReviewedTrips,
      recentReviews
    };

    res.json(stats);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching stats', error: error.message });
  }
};

module.exports = {
  getStats
};
