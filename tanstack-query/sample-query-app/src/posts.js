import React from "react";
import { useQuery } from "@tanstack/react-query";
import  axios from "axios";


const retrievePosts = async () => {
    const response = await axios.get(
      "https://jsonplaceholder.typicode.com/posts",
    );
    return response.data;
  };

export default function Posts() {
    const  {data:posts, isLoading, error }  = useQuery({queryKey:["posts"], queryFn: retrievePosts});
   
    if (isLoading) {
        return <div>Loading...</div>;
    }
    if (error) {
        return <div>Error: {error.message}</div>;
    }

    return (
        <div>
        <h1>Posts</h1>
        {posts?.map(post => (
                <div key={post.id}>
                    <h2>{post.title}</h2>
                    <p>{post.body}</p>
                </div>
            ))}
        </div>
    )
    }
