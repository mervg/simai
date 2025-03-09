# Import the logging system
from logger import (
    initialize_logging,
    VizContext,
    LoggingContext,
    logger
)

def test():
    logger.info("This message should go to S03_test.log")

def viz_test():
    logger.info("This is output from inside the viz_test function")

def main():
    # Initialize logging
    initialize_logging("logging_config.yaml")

    # Test visualization output
    logger.info("Console INFO")
    logger.key("Console KEY")
    with LoggingContext("S01_context"):
        logger.info("1. S01_context")
        with LoggingContext("S02_context"):
            logger.info("1 in S02_context")
        logger.key("2 in S01_context")
        with LoggingContext("S02_context"):
            logger.info("2 in S02_context")
    logger.info("Back to main")
    with LoggingContext("S01_context"):
        logger.info("3 inS01_context")
    with LoggingContext("S02_context"):
        logger.info("3 inS02_context")
    with LoggingContext("S03_test"):
        test()
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    logger.key("Going inViz!")
    with VizContext("F_viz_test"):
        viz_test()
        with VizContext("nested_lines_viz_test"):
            logger.info("Line 1")
            logger.info("Line 2")
        logger.info("Back to F_viz_test")
    logger.key("Back from inViz!")

if __name__ == "__main__":
    main()
