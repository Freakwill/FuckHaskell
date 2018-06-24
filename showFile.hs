#!/usr/local/bin/runHaskell

-- runHaskell opt.hs --filename FILENAME

-- Usage: opt.hs --filename FILENAME [--linen LINES] [-q|--quiet]
--   Print first n lines of a file

import Options.Applicative
import Data.Semigroup ((<>))
import System.IO

takeCont :: Int -> String -> String
takeCont n = unlines . (take n) . lines

data Sample = Sample { filename :: FilePath, linen :: Int, quiet :: Bool}

sample :: Parser Sample
sample = Sample
      <$> strOption
        (long "filename"
        <> short 'f'
        <> metavar "FILENAME"
        <> help "the name of a file")
      <*> option auto
        (long "linen"
        <> short 'n'
        <> help "the number of lines"
        <> showDefault
        <> value 1
        <> metavar "LINES" )
      <*> switch
        (long "quiet"
        <> short 'q'
        <> help "Whether to be quiet" )

main :: IO ()
main = greet =<< execParser opts
  where
    opts = info (sample <**> helper)
      ( fullDesc
     <> progDesc "Print first n lines of a file"
     <> header "filename - my own optparse-applicative" )

greet :: Sample -> IO ()
greet (Sample name n False) = do {
    s <- readFile name;
    putStrLn (takeCont n s)
}
greet _ = return ()
