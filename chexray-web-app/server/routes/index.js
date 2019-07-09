var express = require('express');
var router = express.Router();

router.get('/message', function(req, res, next) {
  res.json('Welcome To Chest X-ray Classifier');
});



module.exports = router;
