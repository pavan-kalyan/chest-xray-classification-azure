const React = require('react')


class Upload extends React.Component {
  constructor(props){
    super(props)
    this.state = {
        text: 'Scan preview',
      file: null,
      file_name: '',
      
    }
    this.handleChange = this.handleChange.bind(this)
    this.addDefaultSrc = this.addDefaultSrc.bind(this)
  }
  addDefaultSrc(ev){
    ev.target.src = require('./sample.png');
  }
  
  handleChange(event) {
      console.log(event.target.files[0])
      
      if(typeof event.target.files[0] === "undefined")
      {
        
        this.refs.input_ref.value='';
         this.setState({
           file: null,

         })
         this.props.file_obj(event.target.files[0])
        return;
      } 
      console.log(event.target.files[0].type);
    console.log("from upload.js "+ event.target.files[0].size);
    if (event.target.files[0].type !== "image/png" ) {
      console.log('not a png.')

      this.props.file_obj(event.target.files[0])
      this.setState({
        file: null,
        file_obj:null,
      })
      this.refs.input_ref.value='';

    }
    else if(event.target.files[0].size/1024 < 1024)
    {
      this.setState({
        file: URL.createObjectURL(event.target.files[0]),
        file_obj:event.target.files[0],
        input_field_value: event.target.files[0].name,
        //text: 'hello'
      }, function () {
          console.log('about to send file '+ this.state.file_obj)
          this.props.file_obj(this.state.file_obj)
      }
      )
    }
    else{
      
      console.log('about to send file in else '+ event.target.files[0])
          this.props.file_obj( event.target.files[0])
          this.setState({
            file: null,
            file_obj:null,
          })
          this.refs.input_ref.value='';
          
    }
    

  }

  render() {
    return (
    <div>
        <h1>{this.state.text}</h1>

      <div>
      <img src={this.state.file} onError={this.addDefaultSrc} width="331" height="331"/>
      </div>
      <div>
        <input type="file" accept="image/png" onChange={this.handleChange} ref="input_ref" />
      </div>
    </div>
    );
  }
}
module.exports = Upload