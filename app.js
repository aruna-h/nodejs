
function sayHello(name){
    console.log('hello '+name);
}
sayHello('aruna');
////////////////////////////////////

var message='hii';
console.log(message);

var message='';
console.log(global.message);        //only available to app.js not for global so are undefined

console.log(module);            //module is not global,in node every file is module and variables,functions defined are scope to that module.they are not available outside that module

////////creating and loading a module/////

const logger=require('./logger');         //while loading a module we need require function
console.log(logger);
//logger.log(message);

const log=require('./logger');
log(message);

////////////////path module//////////////////////////////////

const path=require('path');
var pathobj= path.parse(__filename);
console.log(pathobj);

//////////////////os module///////////////////////////////

const os=require('os');
var totalmemory=os.totalmem();
var freememory=os.freemem();
console.log('total memory'+totalmemory);
console.log('free memory'+freememory);

//template string
//ES6 /ES2015 :ECMAScript 6      -this provides `` to avoid concatenation

console.log(`total memory: ${totalmemory}`);
console.log(`free memory: ${freememory}`);

////////////////////fs module/////////////////////////////////////

const fs=require('fs');
//const files=fs.readdirSync('./');
//console.log(files);

fs.readdir('./',function(err,files){
    if(err) console.log('error ',err);
    else console.log('result '+files);
});
//////////////raise an event using listener//////////////////////////////////

const EventEmitter=require('events');
const emitter=new EventEmitter();

//register a listner
emitter.on('messagelogged',function(){
    console.log('listner called');
});

//release an event
emitter.emit('messagelogged');

//argument events////////////////////////////////////////////////
const EventEmitter=require('events');
const emitter=new EventEmitter();

//register a listner
emitter.on('messagelogged',function(arg){
    console.log('listner called',arg);
});

//release an event
emitter.emit('messagelogged',{ id: 1, url: 'http://' });

///////////////////////////////////////////////////////////////////
const EventEmitter=require('events');
const Logger=require('./logger');
const logger=new Logger();

//register a listner
logger.on('messagelogged',(arg) => {
    console.log('listner called',arg);
});

//release an event
logger.log('message');

///////////////////////////////////////////////////////
const http=require('http');
const server=http.createServer();

server.on('connection',(socket)=>{
    console.log('new connection....');
});

server.listen(3000);

console.log('listening on port 3000..');
