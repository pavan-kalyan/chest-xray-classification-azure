const express = require('express');
const path = require('path');
const logger = require('morgan');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const http = require('http');
var cors = require('cors');

const index = require('./routes/index');
var rp = require('request-promise');

const app = express();

// uncomment after placing your favicon in /public
//app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
//app.use(cors);
app.use(logger('dev'));
app.use(bodyParser.json({limit: "50mb"}));
app.use(bodyParser.urlencoded({limit: "50mb", extended: true, parameterLimit:50000}));
//app.use(bodyParser.json());
//app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'build')));

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');
var request = require('request');
let jsondata = require('./test.json'); 


async function diagnose() {

  
}
app.use('/api', index);
app.post('/diagnose', (req,res) => {
  console.log("diagnose triggered");
  dreq = request.defaults({
    method : "POST",
    followAllRedirects: true,
  });

  
    dreq.post(
      'http://9029851b-945a-4b2d-a756-11da5803f00f.westeurope.azurecontainer.io/score',
      { json: req.body },
      function (error, response, body) {
          if (!error && response.statusCode == 200) {
              console.log('not an error');
              console.log(body);
              //res.json(body);
              //resolve(body);
              //return body;
          }
          else{
            console.log("error");
            console.log(error);
            //res.json(error)
            //reject(error);
            console.log(response.statusMessage);
            //return error;
          }
      }
  ).pipe(res);
  

  

  /*
  dreq.post(
      'http://a91e91a3-57a1-4573-a88e-bc12c26cc59c.westeurope.azurecontainer.io/score',
      { json: jsondata },
      function (error, response, body) {
          if (!error && response.statusCode == 200) {
              console.log('not an error');
              console.log(body);
              res.json(body);
          }
          else{
            console.log("error");
            console.log(error);
            res.json(error)
            console.log(response.statusMessage);
          }
      }
  );
  
  //res.json('post response from deployed server.')

  dreq.on('error', (e) => {
    console.error(`problem with request: ${e.message}`);
  });
  */
});
app.get('*', (req, res) => {
  res.sendFile('build/index.html', { root: global });
});



// catch 404 and forward to error handler
app.use(function(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
