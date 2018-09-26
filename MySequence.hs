module MySequence
( 

)
where


import Control.Monad

-- \m -> mapM . (const m) :: m b -> [a] -> m [b]
myseq :: (Monad m) => m b -> [a] -> m [b]
myseq x = mapM (const x)

repM :: (Monad m) => m a -> Int -> m [a]
repM x n = sequence (replicate n x)

repM_ :: (Monad m) => Int -> m a -> m ()
repM_ n x = sequence_ (replicate n x)

myGet :: String -> IO String
myGet s = do
  putStr s
  getLine

main :: IO ()
main = do 
  x <- myGet "input something:\n"
  repM_ 3 (print x)
  replicateM_ 3 (print (x ++ "again"))


