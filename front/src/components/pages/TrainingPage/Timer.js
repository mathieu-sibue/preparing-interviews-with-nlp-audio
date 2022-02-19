import React, {useEffect} from 'react';
import {Button, Typography} from "@material-ui/core";

function Timer({timeOutHandler, timeOut}){
    const [time, setTime] = React.useState(timeOut)
    const [stop, setStop] = React.useState(false)

    useEffect(() => {
        if(time === 0){
            if(timeOutHandler){
                timeOutHandler();
            }
        }
        else{
            if(!stop){
                const id = setInterval(() => setTime(time - 1), 1000);
                return () => clearInterval(id);
            }
        }
    }, [time]);

    return(
        <Typography>{time} secondes</Typography>
    )
}

export default Timer;