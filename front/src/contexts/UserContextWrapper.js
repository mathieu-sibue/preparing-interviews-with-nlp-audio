import React, { createContext, useState, useEffect } from "react"
import { CircularProgress } from "@material-ui/core"
import userRequests from "../APIrequests/userRequests";


export const UserContext = createContext();

export default function UserContextWrapper({ children }) {


    const [user, setUser] = useState(null);
    const [isLoaded, setIsLoaded] = useState(false);

    const logIn = async (username) => {
        const isLogIn = userRequests.logIn(username)
        const userInfo = {username: username}
        setUser(userInfo)
    }

    const logOut = async () => {
        localStorage.clear()
        setUser(null)
    }

    useEffect(()=>{
        async function retrieveUser(token) {
            if (token !== null) {
                const userInfo = {username : token}
                setUser(userInfo);
            }
            else {
                setUser(null)
            }
        }
        const token = localStorage.getItem("token");
        retrieveUser(token).then(()=>setIsLoaded(true));
    }, [])

    return (
        <div>
            {
                isLoaded?
                    <UserContext.Provider value={{
                        user,
                        setUser,
                        logIn,
                        logOut
                    }}>
                        {children}
                    </UserContext.Provider>:

                    <div style={{display:'flex', justifyContent:'center', alignItems:'center', marginTop: "50vh", transform: "translateY(-50%)"}}>
                        <CircularProgress disableShrink style={{margin: "auto"}}/>
                    </div>
            }
        </div>

    )
}