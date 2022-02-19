//const burl = "https://#####.mvp.#####.fr"
const burl = "http://localhost:5000"

export default {
    logIn : async function(username){
        if(username){
            localStorage.setItem('token', username);
        }
    }
}
