import React, { Component } from 'react';
import logo from './odlogo.svg';
import './App.css';
import Upload from './Upload';
import Spinner from 'react-spinner-material';




const axios = require('axios');


function Table(props)
{
  if(props.dataLoaded)
  {
    return (
      <div >
        <h3> Results</h3>
        <table style={{"borderWidth":"1px", 'borderStyle':'solid','margin':'1em auto'}}>
        <tbody>
          <th>Labels</th>
          <th>Probabilities</th>
        {props.data.labels.map((item, index) => {
             
             return (
                <tr key={index}>
                    <td>{item}</td>
                    <td>{parseFloat(props.data.probabilities[index]*100).toFixed(4)}</td>
                </tr>
              )
           
           })}</tbody>
        </table>
      </div>
    )
  }
  else return null;
}

function List(props) {
  

    if(props.dataLoaded){
      return (
        <div>
          <h3>Results</h3>
          <li >Disease Name - Probabilities</li>
        {props.data.labels.map((item, index) => (
          <li key={index} >{item} - {props.data.probabilities[index]*100}</li>
        ))}
      </div>
      );
    } else {
      return null;
    }
    

}


class App extends Component {
  constructor() {
    super();
    this.handleClick = this.handleClick.bind(this);
    this.handleFile = this.handleFile.bind(this);
    this.handleChangeChk = this.handleChangeChk.bind(this);
    this.getBase64 = this.getBase64.bind(this);
    this.stopspinner = this.stopspinner.bind(this);
    this.state = { 
      chkbox: true,
      message: '' ,
      file_name: '',
      file_obj: null,
      img64str: '',
      loading: false,
      data: null,
      dataLoaded: false,
      file_size: null,
      notification: '',
    };
  }
  handleFile(file) {
    //console.log(file);
    if(typeof file === "undefined")
    {
      console.log(file);
      this.setState({
        file_obj: null,
        
      },() => {return;})
    }
    else if(file.type !== "image/png")
    {
        this.setState({
          notification: 'file seems to be in an incorrect format, please ensure the file is of format type png.',
          file_obj: null,
        },() => {return;})
    }
    else if(file.size/1024 >1024)
    {
      this.setState({
        notification: 'file size is too large!, please ensure file is smaller than 1mb',
        file_obj : null,
      },() => {
          return;
      })
    }
    else{
      this.setState({
        notification: '',
        file_obj : file,
        file_size : file.size,
        file_name : file.name
      })
    }
    
    
    //console.log('file size '+ file.size/1024);
    
    //console.log("file recvd " + URL.createObjectURL(this.state.file_obj))
    //console.log("fine name "+ file.name);
    
  }
  

  componentDidMount() {
    this.setState({message: 'Welcome to Chest X-Ray Classification'})
    /*
    fetch('/api/message')
      .then(response => response.json())
      .then(json => this.setState({ message: json }));
      //console.log(credentials.account_name);
      //execute().then(() => console.log("Done")).catch((e) => console.log(e));
    */
      
  }
  
 
  handleChangeChk(event)
  {
    
    this.setState({chkbox : event.target.checked},() => console.log(this.state.chkbox));
  }
  getBase64(file, cb) {
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
        cb(reader.result)
    };
    reader.onerror = function (error) {
        console.log('Error: ', error);
    };
}

  stopspinner()
  {
    console.log('inside stop spinner')
    this.setState({
      loading: false,
    })
  }
  handleClick(event) {
    console.log('diagnose button clicked');
    console.log(this.state.file_obj)
    if(this.state.file_obj==null)
    {
      console.log("no file selected.")
      this.setState({
        notification: "no image selected",
      })
    }
    else {
    //var file = document.getElementById('fileinput').files[0];
      //var file = this.state.file_obj;

    this.getBase64(this.state.file_obj, (result) => {
  console.log("chckbox state is " + this.state.chkbox);
  result = result.split("base64,")[1];
  this.setState({
    img64str : result,
    loading: true,
    notification: "",
    }, (response) => axios.post('/diagnose', {
      data : this.state.img64str,
      upload : this.state.chkbox,
      file_name : this.state.file_name.match(/^.*?([^\\/.]*)[^\\/]*$/)[1],
    },{headers: {'Content-Type':'application/json','Access-Control-Allow-Origin': '*'}})
      .then( (response) => {
    
          console.log(response.data);
          this.setState({
            data: response.data,
            notification: "",
          },() =>{
            console.log(this.state.data.labels)
            console.log(this.state.data.probabilities)
            this.state.dataLoaded = true;
          })
          
          this.stopspinner();

        })
      .catch( (error) =>{
        this.setState({
          notification: 'There seems to have been an error on the server side.'
        })
          this.stopspinner();
          console.log(error);
}))
  
  

});
  

    } 
    
   
  }


  render() {
    return (
      <div className="App">
        <div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 style={{'padding':'1%','position':'relative','width':'90%'}}>{this.state.message}</h1>
           
        </div>
        <Upload file_obj = {this.handleFile}/>
        
        <p className="App-intro">
          
        </p>
        <div>
        
        <label ><input type="checkbox" defaultChecked={this.state.chkbox}  value={this.state.chkbox} name="permcheckbox" onChange={this.handleChangeChk} />I allow the images uploaded to be used for internal research purposes.</label>
        </div>
        <button onClick={this.handleClick}>
          Diagnose
        </button>
        <p>{this.state.notification}</p>
        <div  style={{display: 'flex',  justifyContent:'center', alignItems:'center',margin:'1em',}}>
        <Spinner size={30} spinnerColor={"#333"} spinnerWidth={1} visible={this.state.loading} />
        </div>
        <Table data={this.state.data} dataLoaded={this.state.dataLoaded}/>
       
      </div>
    );
  }
}

export default App;
