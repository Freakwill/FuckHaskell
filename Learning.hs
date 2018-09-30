
module Learning
( 
   
)
where

loop :: Int -> Int -> [Int] ->[Int] 
loop n i list
    | n <= i = list
    | mod n i == 0 = loop (div n i) 1 (list ++ [i])
    | otherwise = loop n (i+1) list

mystery :: Int -> [Int]
mystery n = loop n 1 []

main :: IO ()
main = do 
   print (mystery 10)