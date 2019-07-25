#!/usr/local/bin/runHaskell

-- runHaskell head.hs --filename FILENAME

-- Usage: opt.hs --filename FILENAME [--linen LINES] [-q|--quiet]
--   Print first n lines of a file

import Options.Applicative
import Data.Semigroup ((<>))
import System.IO

takeCont :: Int -> String -> String
takeCont n = unlines . (take n) . lines

data FileView = FileView { filename :: FilePath, linen :: Int, quiet :: Bool}

fview :: Parser FileView
fview = FileView
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
        <> metavar "LINES")
      <*> switch
        (long "quiet"
        <> short 'q'
        <> help "Whether to be quiet")

-- execParser : Info -> IO FileView
main :: IO ()
main = greet =<< execParser opts
  where
    opts = info (fview <**> helper)
      (fullDesc
     <> progDesc "Print first n lines of a file"
     <> header "filename - my own optparse-applicative")

greet :: FileView -> IO ()
greet (FileView name n False) = do {
    s <- readFile name;
    putStrLn (takeCont n s)
}
greet _ = putStrLn "Be quiet"
